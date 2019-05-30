/*  The MIT License (MIT)
 *  
 *  Copyright (c) 2019 Matias Koskela / Tampere University
 *  Copyright (c) 2018 Kalle Immonen / Tampere University of Technology
 *  
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:
 *  
 *  The above copyright notice and this permission notice shall be included in
 *  all copies or substantial portions of the Software.
 *  
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *  THE SOFTWARE.
 */

#include "bmfr.hpp"
#include "OpenImageIO/imageio.h"
#include "CLUtils/CLUtils.hpp"
#include <functional>
#include <memory>

#define _CRT_SECURE_NO_WARNINGS
#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

// ### Choose your OpenCL device and platform with these defines ###
#define PLATFORM_INDEX 1
#define DEVICE_INDEX 0


// ### Edit these defines if you have different input ###

#define KERNEL_FILENAME "bmfr.cl"

// TODO: turn size and dependent constants into variables (that will be baked as constant inside the kernel)

// TODO detect IMAGE_SIZES automatically from the input files
#define IMAGE_WIDTH 1280
#define IMAGE_HEIGHT 720
// TODO detect FRAME_COUNT from the input files
//#define FRAME_COUNT 1
// Location where input frames and feature buffers are located
#define INPUT_DATA_PATH ../data/classroom/inputs
#define INPUT_DATA_PATH_STR STR(INPUT_DATA_PATH)
// camera_matrices.h is expected to be in the same folder
#include STR(INPUT_DATA_PATH/camera_matrices.h)
// These names are appended with NN.exr, where NN is the frame number
#define NOISY_FILE_NAME INPUT_DATA_PATH_STR"/color"
#define NORMAL_FILE_NAME INPUT_DATA_PATH_STR"/shading_normal"
#define POSITION_FILE_NAME INPUT_DATA_PATH_STR"/world_position"
#define ALBEDO_FILE_NAME INPUT_DATA_PATH_STR"/albedo"
#define OUTPUT_FILE_NAME "outputs/output"


// ### Edit these defines if you want to experiment different parameters ###
// The amount of noise added to feature buffers to cancel sigularities
#define NOISE_AMOUNT 1e-2f

// The amount of new frame used in accumulated frame (1.f would mean no accumulation).
#define BLEND_ALPHA 0.2f
#define SECOND_BLEND_ALPHA 0.1f
#define TAA_BLEND_ALPHA 0.2f

// NOTE: if you want to use other than normal and world_position data you have to make
// it available in the first accumulation kernel and in the weighted sum kernel
#define NOT_SCALED_FEATURE_BUFFERS \
"1.f,"\
"normal.x,"\
"normal.y,"\
"normal.z"

#define USE_SCALED_FEATURES 1

#if USE_SCALED_FEATURES
// The next features are not in the range from -1 to 1 so they are scaled to be from 0 to 1.
#define SCALED_FEATURE_BUFFERS \
",world_position.x,"\
"world_position.y,"\
"world_position.z,"\
"world_position.x*world_position.x,"\
"world_position.y*world_position.y,"\
"world_position.z*world_position.z"
#else
#define SCALED_FEATURE_BUFFERS ""
#endif

// ### Edit these defines to change optimizations for your target hardware ###
// If 1 uses ~half local memory space for R, but computing indexes is more complicated
#define COMPRESSED_R 0

// If 1 stores tmp_data to private memory when it is loaded for dot product calculation
#define CACHE_TMP_DATA 0

// If 1 features_data buffer is in half precision for faster load and store.
// NOTE: if world position values are greater than 256 this cannot be used because
// 256*256 is infinity in half-precision
#define USE_HALF_PRECISION_IN_FEATURES_DATA 0

// If 1 adds __attribute__((reqd_work_group_size(256, 1, 1))) to fitter and
// accumulate_noisy_data kernels. With some codes, attribute made the kernels faster and
// with some it slowed them down.
#define ADD_REQD_WG_SIZE 1

// These local sizes are used with 2D kernels which do not require spesific local size
// (Global sizes are always a multiple of 32)
#define LOCAL_WIDTH 8
#define LOCAL_HEIGHT 8
// Fastest on AMD Radeon Vega Frontier Edition was (LOCAL_WIDTH = 256, LOCAL_HEIGHT = 1)
// Fastest on Nvidia Titan Xp was (LOCAL_WIDTH = 32, LOCAL_HEIGHT = 1)


// ### Do not edit defines after this line unless you know what you are doing ###
// For example, other than 32x32 blocks are not supported
#define BLOCK_EDGE_LENGTH 32

#define BLOCK_PIXELS (BLOCK_EDGE_LENGTH * BLOCK_EDGE_LENGTH)

// Rounds image sizes up to next multiple of BLOCK_EDGE_LENGTH
#define WORKSET_WIDTH  (BLOCK_EDGE_LENGTH * ((IMAGE_WIDTH  + BLOCK_EDGE_LENGTH - 1) / BLOCK_EDGE_LENGTH))
#define WORKSET_HEIGHT (BLOCK_EDGE_LENGTH * ((IMAGE_HEIGHT + BLOCK_EDGE_LENGTH - 1) / BLOCK_EDGE_LENGTH))

// TODO: not sure that the buffers must be OUTPUT_SIZE: IMAGE_WITDH * IMAGE_HEIGHT should be enough
#define OUTPUT_SIZE (WORKSET_WIDTH * WORKSET_HEIGHT)

// 256 is the maximum local size on AMD GCN
// Synchronization within 32x32=1024 block requires unrolling four times
#define LOCAL_SIZE 256

// We need a margin of one block (BLOCK_EDGE_LENGTH) because we are offsetting the blocks
// at most one block to the left or the right to avoid blocky artifacts.
// So most of the time, we have 2 partially filled blocks and BLOCK_COUNT_X-1 full blocks on one row

// Example: image of 8x12 with blocks of 4x1

// No offset: 2x3 = 6 blocks
// |....|....|
// |....|....|
// |....|....|

// With some offset (here (-2, 0)): 3x3 = 9 blocks
// |--..|....|..--|
// |--..|....|..--|
// |--..|....|..--|

// Workset with margin width
#define WORKSET_WITH_MARGINS_WIDTH (WORKSET_WIDTH + BLOCK_EDGE_LENGTH)

// Workset with margin height
#define WORKSET_WITH_MARGINS_HEIGHT (WORKSET_HEIGHT + BLOCK_EDGE_LENGTH)

// Number of blocks in X dim in the workset with margin
#define WORKSET_WITH_MARGIN_BLOCK_COUNT_X (WORKSET_WITH_MARGINS_WIDTH  / BLOCK_EDGE_LENGTH)

// Number of blocks in Y dim in the workset with margin
#define WORKSET_WITH_MARGIN_BLOCK_COUNT_Y (WORKSET_WITH_MARGINS_HEIGHT / BLOCK_EDGE_LENGTH)

// Number of block in the workset with margin
#define WORKSET_WITH_MARGIN_BLOCK_COUNT (WORKSET_WITH_MARGIN_BLOCK_COUNT_X * WORKSET_WITH_MARGIN_BLOCK_COUNT_Y)

// Fitter kernel global range
#define FITTER_KERNEL_GLOBAL_RANGE (LOCAL_SIZE * WORKSET_WITH_MARGIN_BLOCK_COUNT)

// Creates two same buffers and swap() call can be used to change which one is considered
// current and which one previous
template <class T>
class Double_buffer
{
    private:
        T a, b;
        bool swapped;

    public:
		Double_buffer(T aa, T bb): a(aa), b(bb) { }
        template <typename... Args>
        Double_buffer(Args... args) : a(args...), b(args...), swapped(false){};
        T *current() { return swapped ? &a : &b; }
        T *previous() { return swapped ? &b : &a; }
        void swap() { swapped = !swapped; }
};

struct Operation_result
{
    bool success;
    std::string error_message;
    Operation_result(bool success, const std::string &error_message = "") :
        success(success), error_message(error_message) {}
};

Operation_result read_image_file(const std::string &file_name, const int frame, float *buffer)
{
    OpenImageIO::ImageInput *in = OpenImageIO::ImageInput::open(file_name + std::to_string(frame) + ".exr");
    if(!in || in->spec().width != IMAGE_WIDTH || in->spec().height != IMAGE_HEIGHT || in->spec().nchannels != 3)
    {
        return {false, "Can't open image file or it has wrong type: " + file_name};
    }

    // NOTE: this converts .exr files that might be in halfs to single precision floats
    // In the dataset distributed with the BMFR paper all exr files are in single precision
    in->read_image(OpenImageIO::TypeDesc::FLOAT, buffer);
    in->close();

    return {true};
}

Operation_result load_image(cl_float *image, const std::string file_name, const int frame)
{
    Operation_result result = read_image_file(file_name, frame, image);
    if(!result.success)
        return result;

    return {true};
}

float clamp(float value, float minimum, float maximum)
{
    return std::max(std::min(value, maximum), minimum);
}

int bmfr_opencl()
{
    printf("Initialize.\n");
    clutils::CLEnv clEnv;
    cl::Context &context(clEnv.addContext(PLATFORM_INDEX));

    // Find name of the used device
    std::string deviceName;
    clEnv.devices[0][DEVICE_INDEX].getInfo(CL_DEVICE_NAME, &deviceName);
    printf("Using device named: %s\n", deviceName.c_str());

    cl::CommandQueue &queue(clEnv.addQueue(0, DEVICE_INDEX, CL_QUEUE_PROFILING_ENABLE));

    std::string features_not_scaled(NOT_SCALED_FEATURE_BUFFERS);
    std::string features_scaled(SCALED_FEATURE_BUFFERS);
    // + 1 because last one does not have ',' after it.
#if USE_SCALED_FEATURES
	const auto features_not_scaled_count = std::count(features_not_scaled.begin(), features_not_scaled.end(), ',')+1;
	const auto features_scaled_count = std::count(features_scaled.begin(), features_scaled.end(), ',');
#else
	const auto features_not_scaled_count = std::count(features_not_scaled.begin(), features_not_scaled.end(), ',')+1;
	const auto features_scaled_count = 0;
#endif
    // + 3 stands for three noisy spp color channels.
    const auto buffer_count = features_not_scaled_count + features_scaled_count + 3;

    // Create and build the kernel
    std::stringstream build_options;
    build_options <<
        " -D BUFFER_COUNT=" << buffer_count <<
        " -D FEATURES_NOT_SCALED=" << features_not_scaled_count <<
        " -D FEATURES_SCALED=" << features_scaled_count <<
        " -D IMAGE_WIDTH=" << IMAGE_WIDTH <<
        " -D IMAGE_HEIGHT=" << IMAGE_HEIGHT <<
        " -D WORKSET_WIDTH=" << WORKSET_WIDTH <<
        " -D WORKSET_HEIGHT=" << WORKSET_HEIGHT <<
        " -D FEATURE_BUFFERS=" << NOT_SCALED_FEATURE_BUFFERS SCALED_FEATURE_BUFFERS <<
        " -D LOCAL_WIDTH=" << LOCAL_WIDTH <<
        " -D LOCAL_HEIGHT=" << LOCAL_HEIGHT <<
        " -D WORKSET_WITH_MARGINS_WIDTH=" << WORKSET_WITH_MARGINS_WIDTH <<
        " -D WORKSET_WITH_MARGINS_HEIGHT=" << WORKSET_WITH_MARGINS_HEIGHT <<
        " -D BLOCK_EDGE_LENGTH=" << STR(BLOCK_EDGE_LENGTH) <<
        " -D BLOCK_PIXELS=" << BLOCK_PIXELS <<
        " -D R_EDGE=" << buffer_count - 2 <<
        " -D NOISE_AMOUNT=" << STR(NOISE_AMOUNT) <<
        " -D BLEND_ALPHA=" << STR(BLEND_ALPHA) <<
        " -D SECOND_BLEND_ALPHA=" << STR(SECOND_BLEND_ALPHA) <<
        " -D TAA_BLEND_ALPHA=" << STR(TAA_BLEND_ALPHA) <<
        " -D POSITION_LIMIT_SQUARED=" << position_limit_squared <<
        " -D NORMAL_LIMIT_SQUARED=" << normal_limit_squared <<
        " -D COMPRESSED_R=" << STR(COMPRESSED_R) <<
        " -D CACHE_TMP_DATA=" << STR(CACHE_TMP_DATA) <<
        " -D ADD_REQD_WG_SIZE=" << STR(ADD_REQD_WG_SIZE) <<
        " -D LOCAL_SIZE=" << STR(LOCAL_SIZE) <<
        " -D USE_HALF_PRECISION_IN_FEATURES_DATA=" << STR(USE_HALF_PRECISION_IN_FEATURES_DATA);

	// Phase I
	// 3.2 Preprocessing: temporal accumulation of the noisy 1 spp data, which reprojects the previous accumulated data to the new camera frame
    cl::Kernel &accum_noisy_kernel(clEnv.addProgram(0, "bmfr.cl", "accumulate_noisy_data", build_options.str().c_str()));

	// Phase II: feature fitting phase
	// 3.3 Blockwise Multi-Order Feature Regression (BMFR)
	// 3.4 Feature Fitting with Stochastic Regularization
    cl::Kernel &fitter_kernel(clEnv.addProgram(0, "bmfr.cl", "fitter", build_options.str().c_str()));

    cl::Kernel &weighted_sum_kernel(clEnv.addProgram(0, "bmfr.cl", "weighted_sum", build_options.str().c_str()));
    cl::Kernel &accum_filtered_kernel(clEnv.addProgram(0, "bmfr.cl", "accumulate_filtered_data", build_options.str().c_str()));
    cl::Kernel &taa_kernel(clEnv.addProgram(0, "bmfr.cl", "taa", build_options.str().c_str()));

    cl::NDRange k_workset_with_margin_global_size(WORKSET_WITH_MARGINS_WIDTH, WORKSET_WITH_MARGINS_HEIGHT);
    cl::NDRange k_workset_global_size(WORKSET_WIDTH, WORKSET_HEIGHT);
    cl::NDRange k_local_size(LOCAL_WIDTH, LOCAL_HEIGHT);
    cl::NDRange k_fitter_global_size(FITTER_KERNEL_GLOBAL_RANGE);
    cl::NDRange k_fitter_local_size(LOCAL_SIZE);

    // Load input data arrays from disk to host memory
    printf("Loading input data.\n");
    std::vector<cl_float> out_data[FRAME_COUNT];
    std::vector<cl_float> albedos[FRAME_COUNT];
    std::vector<cl_float> normals[FRAME_COUNT];
    std::vector<cl_float> positions[FRAME_COUNT];
    std::vector<cl_float> noisy_input[FRAME_COUNT];
    bool error = false;
	#pragma omp parallel for
    for(int frame = 0; frame < FRAME_COUNT; ++frame)
    {
        if(error)
            continue;

        out_data[frame].resize(3 * OUTPUT_SIZE);

        albedos[frame].resize(3 * IMAGE_WIDTH * IMAGE_HEIGHT);
        Operation_result result = load_image(albedos[frame].data(), ALBEDO_FILE_NAME,
            frame);
        if(!result.success)
        {
            error = true;
            printf("Albedo buffer loading failed, reason: %s\n",
                   result.error_message.c_str());
            continue;
        }

        normals[frame].resize(3 * IMAGE_WIDTH * IMAGE_HEIGHT);
        result = load_image(normals[frame].data(), NORMAL_FILE_NAME, frame);
        if(!result.success)
        {
            error = true;
            printf("Normal buffer loading failed, reason: %s\n",
                   result.error_message.c_str());
            continue;
        }

        positions[frame].resize(3 * IMAGE_WIDTH * IMAGE_HEIGHT);
        result = load_image(positions[frame].data(), POSITION_FILE_NAME, frame);
        if(!result.success)
        {
            error = true;
            printf("Position buffer loading failed, reason: %s\n",
                   result.error_message.c_str());
            continue;
        }

        noisy_input[frame].resize(3 * IMAGE_WIDTH * IMAGE_HEIGHT);
        result = load_image(noisy_input[frame].data(), NOISY_FILE_NAME, frame);
        if(!result.success)
        {
            error = true;
            printf("Position buffer loading failed, reason: %s\n",
                   result.error_message.c_str());
            continue;
        }
    }

    if(error)
    {
        printf("One or more errors occurred during buffer loading\n");
        return 1;
    }

    // Create OpenCL buffers

	// (World) normals buffers (3 * float32) in [-1, +1]^3
	// TODO: compress data? half? -> would require different storage than feature buffer
    Double_buffer<cl::Buffer> normals_buffer(context, CL_MEM_READ_WRITE, IMAGE_WIDTH * IMAGE_HEIGHT  * 3 * sizeof(cl_float));

	// World positions buffer (3 * float32)
	// TODO: normalize in [0, 1] (or [-1, +1])
    Double_buffer<cl::Buffer> positions_buffer(context, CL_MEM_READ_WRITE, IMAGE_WIDTH * IMAGE_HEIGHT  * 3 * sizeof(cl_float));
    
	// Noisy 1spp color buffer (3 * float32)
	Double_buffer<cl::Buffer> noisy_1spp_buffer(context, CL_MEM_READ_WRITE, IMAGE_WIDTH * IMAGE_HEIGHT  * 3 * sizeof(cl_float));

	// Features buffer size
    const size_t features_buffer_datatype_size = USE_HALF_PRECISION_IN_FEATURES_DATA ? sizeof(cl_half) : sizeof(cl_float);
	const size_t features_buffer_size = WORKSET_WITH_MARGINS_WIDTH * WORKSET_WITH_MARGINS_HEIGHT * buffer_count * features_buffer_datatype_size;

	// Features buffer (half or single-precision) (3 * float16 or 3 * float32)
    cl::Buffer features_buffer(context, CL_MEM_READ_WRITE, features_buffer_size, nullptr);

	// Noise-free color estimate (3 * float32)
    cl::Buffer noisefree_1spp(context, CL_MEM_READ_WRITE, OUTPUT_SIZE * 3 * sizeof(cl_float));

	// TODO: why does the size of this buffer is WORKSET_WITH_MARGINS_WIDTH x WORKSET_WITH_MARGINS_HEIGHT and not WORKSET_WIDTH x WORKSET_HEIGHT?
	// -> it seems we are not writing/reading in the part of the buffer that is outside of image (because of offsets)
	// (see kernel 'accumulate_filtered_data' that is the only kernel using it)
	// --> should have the same size as 'noisefree_1spp_acc_tonemapped'
	// Noise-free accumulated color estimate (3 * float32)
    Double_buffer<cl::Buffer> noisefree_1spp_accumulated(context, CL_MEM_READ_WRITE, WORKSET_WITH_MARGINS_WIDTH * WORKSET_WITH_MARGINS_HEIGHT * 3 * sizeof(cl_float));

	// Final antialiased color buffer (3 * float32)
    Double_buffer<cl::Buffer> result_buffer(context, CL_MEM_READ_WRITE, OUTPUT_SIZE * 3 * sizeof(cl_float));

	// Previous frame pixel coordinates (after reprojection) (2 * float32)
    cl::Buffer prev_frame_pixel_coords_buffer(context, CL_MEM_READ_WRITE, OUTPUT_SIZE * sizeof(cl_float2));

	// Validity mask of reprojected bilinear samples into previous frame (uchar 8bits)
    cl::Buffer prev_frame_bilinear_samples_validity_mask(context, CL_MEM_READ_WRITE, OUTPUT_SIZE * sizeof(cl_uchar));

	// Albedo buffer (3 * float32) // TODO: compress this
    cl::Buffer albedo_buffer(context, CL_MEM_READ_ONLY, IMAGE_WIDTH * IMAGE_HEIGHT * 3 * sizeof(cl_float));

	// Tonemapped noise-free color estimate w/ albedo (3 * float32)
    cl::Buffer noisefree_1spp_acc_tonemapped(context, CL_MEM_READ_WRITE, IMAGE_WIDTH * IMAGE_HEIGHT * 3 * sizeof(cl_float));

	// Features weights per color channel (x3) (computed by the BMFR) (3 * float32)
    //cl::Buffer features_weights_buffer(context, CL_MEM_READ_WRITE, (FITTER_KERNEL_GLOBAL_RANGE / 256) * (buffer_count - 3) * 3 * sizeof(cl_float));
    cl::Buffer features_weights_buffer(context, CL_MEM_READ_WRITE, WORKSET_WITH_MARGIN_BLOCK_COUNT * (buffer_count - 3) * 3 * sizeof(cl_float));

	// Min and max of features values per block (world_positions) (6 * 2 * float32)
    //cl::Buffer features_min_max_buffer(context, CL_MEM_READ_WRITE, (FITTER_KERNEL_GLOBAL_RANGE / 256) * 6 * sizeof(cl_float2));
    cl::Buffer features_min_max_buffer(context, CL_MEM_READ_WRITE, WORKSET_WITH_MARGIN_BLOCK_COUNT * 6 * sizeof(cl_float2));

	// Number of samples accumulated (for cumulative moving average) (char 8bits)
    Double_buffer<cl::Buffer> spp_buffer(context, CL_MEM_READ_WRITE, OUTPUT_SIZE  * sizeof(cl_char));

	std::vector<Double_buffer<cl::Buffer> *> all_double_buffers =
	{
		&normals_buffer,
		&positions_buffer,
		&noisy_1spp_buffer,
		&noisefree_1spp_accumulated,
		&result_buffer,
		&spp_buffer
	};

    // Set kernel arguments
    int arg_index = 0;
    accum_noisy_kernel.setArg(arg_index++, prev_frame_pixel_coords_buffer);				// [out] Previous frame pixel coordinates (after reprojection)
    accum_noisy_kernel.setArg(arg_index++, prev_frame_bilinear_samples_validity_mask);	// [out] Validity mask of bilinear samples in previous frame (after reprojection) (i.e valid reprojection = no disoclusion or outside frame)

    arg_index = 0;
	
	#if COMPRESSED_R
	// TODO: replace 'buffer_count-2'
	// Explanations: The number of features M is buffer_count - 3 because buffer_count comprises the 3 noisy 1spp color channels inputs.
	// To this we add 1 because we concatenate z(c) which is the c channel of the noisy path-traced input with makes a size of 'buffer_count-2'.
	// "The Householder QR factorization yields an (M + 1)x(M + 1) upper triangular matrix R(c)"
	// See section 3.3 and 3.4.

	// Computed via sum of arithmetic sequence (that for the upper right triangle):
	//    0  1  2  3  4  5 x
	// 0 00 01 02 03 04 05
	// 1  - 11 12 13 14 15
	// 2  -  - 22 23 24 25
	// 3  -  -  - 33 34 35
	// 4  -  -  -  - 44 45
	// 5  -  -  -  -  - 55
	// y
	// (1 + 2 + ... + buffer_count - 2) = (buffer_count-2+1)*(buffer_count-2)/2 = (buffer_count-1)*(buffer_count-2)/2
    const auto r_size = ((buffer_count - 2) * (buffer_count - 1) / 2) * sizeof(cl_float3);
	#else
    const auto r_size = (buffer_count - 2) * (buffer_count - 2) * sizeof(cl_float3);
	#endif

	// https://www.khronos.org/registry/OpenCL/sdk/1.1/docs/man/xhtml/clSetKernelArg.html
	// Note: For arguments declared with the __local qualifier, the size specified will be the size in bytes of the buffer that must be allocated for the __local argument.
    fitter_kernel.setArg(arg_index++, LOCAL_SIZE * sizeof(float), nullptr);		// [local] Size of the shared memory used to perform parrallel reduction (max, min, sum)
    fitter_kernel.setArg(arg_index++, BLOCK_PIXELS * sizeof(float), nullptr);	// [local] Shared memory used to store the 'u' vectors
    fitter_kernel.setArg(arg_index++, r_size, nullptr);							// [local] Shared memory used to store the R matrix of the QR factorization
    fitter_kernel.setArg(arg_index++, features_weights_buffer);					// [out]   Features weights
    fitter_kernel.setArg(arg_index++, features_min_max_buffer);					// [out]   Min and max of features values per block (world_positions)

    arg_index = 0;
    weighted_sum_kernel.setArg(arg_index++, features_weights_buffer);	// [in]	 Features weights computed by the fitter kernel
    weighted_sum_kernel.setArg(arg_index++, features_min_max_buffer);	// [in]  Min and max of features values per block (world_positions)
    weighted_sum_kernel.setArg(arg_index++, noisefree_1spp);	// [out] Noise-free color estimate

    arg_index = 0;
    accum_filtered_kernel.setArg(arg_index++, noisefree_1spp);					// [in]  Noise free color estimate (computed as the weighted sum of the features)
    accum_filtered_kernel.setArg(arg_index++, prev_frame_pixel_coords_buffer);				// [in]  Previous frame pixel coordinates (after reprojection)
    accum_filtered_kernel.setArg(arg_index++, prev_frame_bilinear_samples_validity_mask);	// [in]  Validity mask of bilinear samples in previous frame (after reprojection)
    accum_filtered_kernel.setArg(arg_index++, albedo_buffer);								// [in]  Albedo buffer of the current frame (non-noisy)
    accum_filtered_kernel.setArg(arg_index++, noisefree_1spp_acc_tonemapped);							// [out] Accumulated and tonemapped noise-free color estimate

    arg_index = 0;
    taa_kernel.setArg(arg_index++, prev_frame_pixel_coords_buffer);	// [in] Previous frame pixel coordinates (after reprojection)
    taa_kernel.setArg(arg_index++, noisefree_1spp_acc_tonemapped);				// [in]	Current frame color buffer
    queue.finish();

    std::vector<clutils::GPUTimer<std::milli>> accum_noisy_timer;
    std::vector<clutils::GPUTimer<std::milli>> copy_timer;
    std::vector<clutils::GPUTimer<std::milli>> fitter_timer;
    std::vector<clutils::GPUTimer<std::milli>> weighted_sum_timer;
    std::vector<clutils::GPUTimer<std::milli>> accum_filtered_timer;
    std::vector<clutils::GPUTimer<std::milli>> taa_timer;
    accum_noisy_timer.assign(FRAME_COUNT - 1, clutils::GPUTimer<std::milli>(clEnv.devices[0][0]));
    copy_timer.assign(FRAME_COUNT, clutils::GPUTimer<std::milli>(clEnv.devices[0][0]));
    fitter_timer.assign(FRAME_COUNT, clutils::GPUTimer<std::milli>(clEnv.devices[0][0]));
    weighted_sum_timer.assign(FRAME_COUNT, clutils::GPUTimer<std::milli>(clEnv.devices[0][0]));
    accum_filtered_timer.assign(FRAME_COUNT - 1, clutils::GPUTimer<std::milli>(clEnv.devices[0][0]));
    taa_timer.assign(FRAME_COUNT - 1, clutils::GPUTimer<std::milli>(clEnv.devices[0][0]));

    clutils::ProfilingInfo<FRAME_COUNT - 1> profile_info_accum_noisy("Accumulation of noisy data");
    clutils::ProfilingInfo<FRAME_COUNT>		profile_info_copy("Copy input buffer");
    clutils::ProfilingInfo<FRAME_COUNT>		profile_info_fitter("Fitting feature buffers to noisy data");
    clutils::ProfilingInfo<FRAME_COUNT>		profile_info_weighted_sum("Weighted sum");
    clutils::ProfilingInfo<FRAME_COUNT - 1> profile_info_accum_filtered("Accumulation of filtered data");
    clutils::ProfilingInfo<FRAME_COUNT - 1> profile_info_taa("TAA");
    clutils::ProfilingInfo<FRAME_COUNT - 1> profile_info_total("Total time in all kernels (including intermediate launch overheads)");

    printf("Run and profile kernels.\n");

	// Note: enqueueNDRangeKernel takes in the global_size and the local_size.
	// In CUDA, a dispatch takes in the grid_size and the block_size.
	// The correspondance is as follow:
	//	global_size = grid_size * block_size
	//	local_size = block_size

    // Note: in real use case there would not be WriteBuffer and ReadBuffer function calls
    // because the input data comes from the path tracer and output goes to the screen
    for(int frame = 0; frame < FRAME_COUNT; ++frame)
    {
		const cl_bool blocking_write = true;
		// https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/clEnqueueWriteBuffer.html
		// Enqueue commands to write to a buffer object from host memory (= cudaMemcpy(..., cudaMemcpyHostToDevice))
        queue.enqueueWriteBuffer(albedo_buffer, blocking_write, 0, IMAGE_WIDTH * IMAGE_HEIGHT * 3 * sizeof(cl_float), albedos[frame].data());
        queue.enqueueWriteBuffer(*normals_buffer.current(), blocking_write, 0, IMAGE_WIDTH * IMAGE_HEIGHT * 3 * sizeof(cl_float), normals[frame].data());
        queue.enqueueWriteBuffer(*positions_buffer.current(), blocking_write, 0, IMAGE_WIDTH * IMAGE_HEIGHT * 3 * sizeof(cl_float), positions[frame].data());
        queue.enqueueWriteBuffer(*noisy_1spp_buffer.current(), blocking_write, 0, IMAGE_WIDTH * IMAGE_HEIGHT * 3 * sizeof(cl_float), noisy_input[frame].data());

		// Phase I:
		//  - accumulate noisy 1spp
		//  - compute previous frame pixel coordinates (after reprojection)
		//  - generate validity bit mask of bilinear samples of previous frame
		//  - concatenate the different features in a single buffer
        // Note: On the first frame accum_noisy_kernel just copies to the features_buffer
        arg_index = 2;
        accum_noisy_kernel.setArg(arg_index++, *normals_buffer.current());			// [in]  Current  (world) normals
        accum_noisy_kernel.setArg(arg_index++, *normals_buffer.previous());			// [in]  Previous (world) normals
        accum_noisy_kernel.setArg(arg_index++, *positions_buffer.current());		// [in]  Current  world positions
        accum_noisy_kernel.setArg(arg_index++, *positions_buffer.previous());		// [in]  Previous world positions
        accum_noisy_kernel.setArg(arg_index++, *noisy_1spp_buffer.current());	// [out] Current  noisy 1spp color
        accum_noisy_kernel.setArg(arg_index++, *noisy_1spp_buffer.previous());// [in]  Previous noisy 1spp color
        accum_noisy_kernel.setArg(arg_index++, *spp_buffer.previous());				// [in]  Previous number of samples accumulated (for CMA)
        accum_noisy_kernel.setArg(arg_index++, *spp_buffer.current());				// [out] Current  number of samples accumulated (for CMA)
        accum_noisy_kernel.setArg(arg_index++, features_buffer);					// [out] Features buffer (half or single-precision)
        const int matrix_index = frame == 0 ? 0 : frame - 1;
        accum_noisy_kernel.setArg(arg_index++, sizeof(cl_float16), &(camera_matrices[matrix_index][0][0])); // [in] ViewProj matrix of previous frame
        accum_noisy_kernel.setArg(arg_index++, sizeof(cl_float2), &(pixel_offsets[frame][0]));
        accum_noisy_kernel.setArg(arg_index++, sizeof(cl_int), &frame); // [in] Current frame number
        queue.enqueueNDRangeKernel(accum_noisy_kernel, cl::NullRange, k_workset_with_margin_global_size, k_local_size, nullptr, &accum_noisy_timer[matrix_index].event());

		// Phase II: Blockwise Multi-Order Feature Regression (BMFR)
		// -> compute features weightss
        arg_index = 5;
        fitter_kernel.setArg(arg_index++, features_buffer);			// [in] Features buffer (half or single-precision)
        fitter_kernel.setArg(arg_index++, sizeof(cl_int), &frame);  // [in] Current frame number
        queue.enqueueNDRangeKernel(fitter_kernel, cl::NullRange, k_fitter_global_size, k_fitter_local_size, nullptr, &fitter_timer[frame].event());

		// Phase II: Compute noise free color estimate (weighted sum of features)
        arg_index = 3;
        weighted_sum_kernel.setArg(arg_index++, *normals_buffer.current());			 // [in] Current (world) normals
        weighted_sum_kernel.setArg(arg_index++, *positions_buffer.current());		 // [in] Current world positions
        weighted_sum_kernel.setArg(arg_index++, *noisy_1spp_buffer.current()); // [in] Current noisy 1spp color (only used for debugging)
        weighted_sum_kernel.setArg(arg_index++, sizeof(cl_int), &frame);			 // [in] Current frame number
        queue.enqueueNDRangeKernel(weighted_sum_kernel, cl::NullRange, k_workset_global_size, k_local_size, nullptr, &weighted_sum_timer[frame].event());

		// Phase III: Postprocessing
		// -> accumulate noise-free color estimate + output a tonemapped version
        arg_index = 5;
        accum_filtered_kernel.setArg(arg_index++, *spp_buffer.current());	// [in]	 Current number of samples accumulated (for CMA)
        accum_filtered_kernel.setArg(arg_index++, *noisefree_1spp_accumulated.previous()); // [in]  Previous frame noise-free accumulated color estimate 
        accum_filtered_kernel.setArg(arg_index++, *noisefree_1spp_accumulated.current());	 // [out] Current frame noise-free accumulated color estimate
        accum_filtered_kernel.setArg(arg_index++, sizeof(cl_int), &frame);	// [in]  Current frame number
        queue.enqueueNDRangeKernel(accum_filtered_kernel, cl::NullRange, k_workset_global_size, k_local_size, nullptr, &accum_filtered_timer[matrix_index].event());

		// Phase III: Temporal antialiasing
        arg_index = 2;
        taa_kernel.setArg(arg_index++, *result_buffer.current());	// [out] Antialiased frame color buffer
        taa_kernel.setArg(arg_index++, *result_buffer.previous());	// [in]  Previous frame color buffer
        taa_kernel.setArg(arg_index++, sizeof(cl_int), &frame);		// [in]  Current frame number
        queue.enqueueNDRangeKernel(taa_kernel, cl::NullRange, k_workset_global_size, k_local_size, nullptr, &taa_timer[matrix_index].event());

        // This is not timed because in real use case the result is stored to frame buffer
        queue.enqueueReadBuffer(*result_buffer.current(), false, 0, OUTPUT_SIZE * 3 * sizeof(cl_float), out_data[frame].data());

        // Swap all double buffers
        std::for_each(all_double_buffers.begin(), all_double_buffers.end(), std::bind(&Double_buffer<cl::Buffer>::swap, std::placeholders::_1));
    }
    queue.finish();

    // Store profiling data
    for(int i = 0; i < FRAME_COUNT; ++i)
    {
        if(i > 0)
        {
            profile_info_accum_noisy[i - 1] = accum_noisy_timer[i - 1].duration();
            profile_info_accum_filtered[i - 1] = accum_filtered_timer[i - 1].duration();
            profile_info_taa[i - 1] = taa_timer[i - 1].duration();

            cl_ulong total_start =
                accum_noisy_timer[i - 1].event().getProfilingInfo<CL_PROFILING_COMMAND_START>();
            cl_ulong total_end =
                taa_timer[i - 1].event().getProfilingInfo<CL_PROFILING_COMMAND_END>();
            profile_info_total[i - 1] =
                (total_end - total_start) * taa_timer[i - 1].getUnit();
        }
        profile_info_fitter[i] = fitter_timer[i].duration();
        profile_info_weighted_sum[i] = weighted_sum_timer[i].duration();
    }

    if(FRAME_COUNT > 1)
        profile_info_accum_noisy.print();
    profile_info_fitter.print();
    profile_info_weighted_sum.print();
    if(FRAME_COUNT > 1)
    {
        profile_info_accum_filtered.print();
        profile_info_taa.print();
        profile_info_total.print();
    }

    // Store results
    error = false;
	#pragma omp parallel for
    for(int frame = 0; frame < FRAME_COUNT; ++frame)
    {
        if(error)
            continue;

        // Output image
        std::string output_file_name = OUTPUT_FILE_NAME + std::to_string(frame) + ".png";
        // Crops back from WORKSET_SIZE to IMAGE_SIZE
        OpenImageIO::ImageSpec spec(IMAGE_WIDTH, IMAGE_HEIGHT, 3,
                                    OpenImageIO::TypeDesc::FLOAT);
        std::unique_ptr<OpenImageIO::ImageOutput>
            out(OpenImageIO::ImageOutput::create(output_file_name));
        if(out && out->open(output_file_name, spec))
        {
            out->write_image(OpenImageIO::TypeDesc::FLOAT, out_data[frame].data(),
                             3 * sizeof(cl_float), WORKSET_WIDTH * 3 * sizeof(cl_float), 0);
            out->close();
        }
        else
        {
            printf("Can't create image file on disk to location %s\n",
                   output_file_name.c_str());
            error = true;
            continue;
        }
    }

    if(error)
    {
        printf("One or more errors occurred during image saving\n");
        return 1;
    }

    return 0;
}


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
 
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
 
#define MAX_SOURCE_SIZE (0x100000)

int bmfr_c_opencl(TmpData & tmpData)
{
    printf("Initialize.\n");

	// Based on: https://gist.github.com/courtneyfaulkner/7919509
	{
		int i, j;
		char* info;
		size_t infoSize;
		char* value;
		size_t valueSize;
		cl_uint platformCount;
		cl_platform_id* platforms;
		cl_uint deviceCount;
		cl_device_id* devices;
		cl_uint maxComputeUnits;
		const char* attributeNames[5] = { "Name", "Vendor", "Version", "Profile", "Extensions" };
		const cl_platform_info attributeTypes[5] = { CL_PLATFORM_NAME, CL_PLATFORM_VENDOR, CL_PLATFORM_VERSION, CL_PLATFORM_PROFILE, CL_PLATFORM_EXTENSIONS };
		const int attributeCount = sizeof(attributeNames) / sizeof(char*);

		// get all platforms
		clGetPlatformIDs(5, NULL, &platformCount);
		platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * platformCount);
		clGetPlatformIDs(platformCount, platforms, NULL);

		for (i = 0; i < platformCount; i++)
		{
			printf("\n %d. Platform \n", i+1);

			for (j = 0; j < attributeCount; j++) {

				// get platform attribute value size
				clGetPlatformInfo(platforms[i], attributeTypes[j], 0, NULL, &infoSize);
				info = (char*) malloc(infoSize);

				// get platform attribute value
				clGetPlatformInfo(platforms[i], attributeTypes[j], infoSize, info, NULL);

				printf("  %d.%d %-11s: %s\n", i+1, j+1, attributeNames[j], info);
				free(info);
			}

			printf("\n");

			// get all devices
			clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
			devices = (cl_device_id*) malloc(sizeof(cl_device_id) * deviceCount);
			clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL);

			// for each device print critical attributes
			for (j = 0; j < deviceCount; j++) {

				// print device name
				clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 0, NULL, &valueSize);
				value = (char*) malloc(valueSize);
				clGetDeviceInfo(devices[j], CL_DEVICE_NAME, valueSize, value, NULL);
				printf("%d. Device: %s\n", j+1, value);
				free(value);

				// print hardware device version
				clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, 0, NULL, &valueSize);
				value = (char*) malloc(valueSize);
				clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, valueSize, value, NULL);
				printf(" %d.%d Hardware version: %s\n", j+1, 1, value);
				free(value);

				// print software driver version
				clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, 0, NULL, &valueSize);
				value = (char*) malloc(valueSize);
				clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, valueSize, value, NULL);
				printf(" %d.%d Software version: %s\n", j+1, 2, value);
				free(value);

				// print c version supported by compiler for device
				clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &valueSize);
				value = (char*) malloc(valueSize);
				clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, valueSize, value, NULL);
				printf(" %d.%d OpenCL C version: %s\n", j+1, 3, value);
				free(value);

				// print parallel compute units
				clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS,
						sizeof(maxComputeUnits), &maxComputeUnits, NULL);
				printf(" %d.%d Parallel compute units: %d\n", j+1, 4, maxComputeUnits);

			}

			free(devices);

		}

		free(platforms);
	}

#if 1
	cl_device_id device_id = NULL;
	cl_uint ret_num_devices;
	cl_uint ret_num_platforms;
	cl_int ret = clGetPlatformIDs(0, NULL, &ret_num_platforms);
	cl_platform_id *platforms = NULL;
	platforms = (cl_platform_id*)malloc(ret_num_platforms*sizeof(cl_platform_id));
	ret = clGetPlatformIDs(ret_num_platforms, platforms, NULL);
	assert(ret == CL_SUCCESS);
	ret = clGetDeviceIDs(platforms[1], CL_DEVICE_TYPE_ALL, 1, &device_id, &ret_num_devices);
	assert(ret == CL_SUCCESS);
	
	// Print device name
	char* value;
    size_t valueSize;
    clGetDeviceInfo(device_id, CL_DEVICE_NAME, 0, NULL, &valueSize);
    value = (char*) malloc(valueSize);
    clGetDeviceInfo(device_id, CL_DEVICE_NAME, valueSize, value, NULL);
    printf("\nSelected device: %s\n", value);
    free(value);
	free(platforms);
	platforms = nullptr;
#else
	// Get platform and device information
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;   
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
	assert(ret == CL_SUCCESS);
    
	ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);
	assert(ret == CL_SUCCESS);

	// Print device name
    size_t valueSize;
    clGetDeviceInfo(device_id, CL_DEVICE_NAME, 0, NULL, &valueSize);
    char * value = (char*) malloc(valueSize);
    clGetDeviceInfo(device_id, CL_DEVICE_NAME, valueSize, value, NULL);
    printf("Selected device: %s\n", value);
    free(value);
#endif

	// Create an OpenCL context
	cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
 
	// Create a command queue
	cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
 

	std::string features_not_scaled(NOT_SCALED_FEATURE_BUFFERS);
	std::string features_scaled(SCALED_FEATURE_BUFFERS);
	// + 1 because last one does not have ',' after it.
#if USE_SCALED_FEATURES
	const auto features_not_scaled_count = std::count(features_not_scaled.begin(), features_not_scaled.end(), ',')+1;
	const auto features_scaled_count = std::count(features_scaled.begin(), features_scaled.end(), ',');
#else
	const auto features_not_scaled_count = std::count(features_not_scaled.begin(), features_not_scaled.end(), ',')+1;
	const auto features_scaled_count = 0;
#endif
	// + 3 stands for three noisy spp color channels.
	const auto buffer_count = features_not_scaled_count + features_scaled_count + 3;

	// Create and build the kernel
	std::stringstream build_options;
	build_options <<
		" -D BUFFER_COUNT=" << buffer_count <<
		" -D FEATURES_NOT_SCALED=" << features_not_scaled_count <<
		" -D FEATURES_SCALED=" << features_scaled_count <<
		" -D IMAGE_WIDTH=" << IMAGE_WIDTH <<
		" -D IMAGE_HEIGHT=" << IMAGE_HEIGHT <<
		" -D WORKSET_WIDTH=" << WORKSET_WIDTH <<
		" -D WORKSET_HEIGHT=" << WORKSET_HEIGHT <<
		" -D FEATURE_BUFFERS=" << NOT_SCALED_FEATURE_BUFFERS SCALED_FEATURE_BUFFERS <<
		" -D LOCAL_WIDTH=" << LOCAL_WIDTH <<
		" -D LOCAL_HEIGHT=" << LOCAL_HEIGHT <<
		" -D WORKSET_WITH_MARGINS_WIDTH=" << WORKSET_WITH_MARGINS_WIDTH <<
		" -D WORKSET_WITH_MARGINS_HEIGHT=" << WORKSET_WITH_MARGINS_HEIGHT <<
		" -D BLOCK_EDGE_LENGTH=" << STR(BLOCK_EDGE_LENGTH) <<
		" -D BLOCK_PIXELS=" << BLOCK_PIXELS <<
		" -D R_EDGE=" << buffer_count - 2 <<
		" -D NOISE_AMOUNT=" << STR(NOISE_AMOUNT) <<
		" -D BLEND_ALPHA=" << STR(BLEND_ALPHA) <<
		" -D SECOND_BLEND_ALPHA=" << STR(SECOND_BLEND_ALPHA) <<
		" -D TAA_BLEND_ALPHA=" << STR(TAA_BLEND_ALPHA) <<
		" -D POSITION_LIMIT_SQUARED=" << position_limit_squared <<
		" -D NORMAL_LIMIT_SQUARED=" << normal_limit_squared <<
		" -D COMPRESSED_R=" << STR(COMPRESSED_R) <<
		" -D CACHE_TMP_DATA=" << STR(CACHE_TMP_DATA) <<
		" -D ADD_REQD_WG_SIZE=" << STR(ADD_REQD_WG_SIZE) <<
		" -D LOCAL_SIZE=" << STR(LOCAL_SIZE) <<
		" -D USE_HALF_PRECISION_IN_FEATURES_DATA=" << STR(USE_HALF_PRECISION_IN_FEATURES_DATA);

	// Load the kernel source code into the array source_str
	FILE *fp;
	char *source_str;
	size_t source_size;
 
	fp = fopen(KERNEL_FILENAME, "r");
	if(!fp)
	{
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
    
	char * file_content = (char*)malloc(MAX_SOURCE_SIZE);

	source_str  = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);

	// Create a program from the kernel source
	cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);

	// Build the program
	ret = clBuildProgram(program, 1, &device_id, build_options.str().c_str(), NULL, NULL);

	if(ret != CL_SUCCESS)
	{
		size_t len = 0;
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
		char *buffer = (char*)malloc(len * sizeof(char));
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, len, buffer, NULL);

		printf("Compilation failed: %s\n", buffer);
	}

	assert(ret == CL_SUCCESS);

	// Phase I
	// 3.2 Preprocessing: temporal accumulation of the noisy 1 spp data, which reprojects the previous accumulated data to the new camera frame
	//cl::Kernel &accum_noisy_kernel(clEnv.addProgram(0, "bmfr.cl", "accumulate_noisy_data", build_options.str().c_str()));
	cl_kernel accum_noisy_kernel = clCreateKernel(program, "accumulate_noisy_data", &ret);
	assert(ret == CL_SUCCESS);

	// Phase II: feature fitting phase
	// 3.3 Blockwise Multi-Order Feature Regression (BMFR)
	// 3.4 Feature Fitting with Stochastic Regularization
	cl_kernel fitter_kernel = clCreateKernel(program, "fitter", &ret);
	assert(ret == CL_SUCCESS);

	cl_kernel weighted_sum_kernel = clCreateKernel(program, "weighted_sum", &ret);
	assert(ret == CL_SUCCESS);

	cl_kernel accum_filtered_kernel = clCreateKernel(program, "accumulate_filtered_data", &ret);
	assert(ret == CL_SUCCESS);

	cl_kernel taa_kernel = clCreateKernel(program, "taa", &ret);
	assert(ret == CL_SUCCESS);

	// Load input data arrays from disk to host memory
	printf("Loading input data.\n");
	std::vector<cl_float> out_data[FRAME_COUNT];
	std::vector<cl_float> albedos[FRAME_COUNT];
	std::vector<cl_float> normals[FRAME_COUNT];
	std::vector<cl_float> positions[FRAME_COUNT];
	std::vector<cl_float> noisy_input[FRAME_COUNT];
	bool error = false;
	#pragma omp parallel for
	for(int frame = 0; frame < FRAME_COUNT; ++frame)
	{
		if(error)
			continue;

		out_data[frame].resize(3 * OUTPUT_SIZE);

		albedos[frame].resize(3 * IMAGE_WIDTH * IMAGE_HEIGHT);
		Operation_result result = load_image(albedos[frame].data(), ALBEDO_FILE_NAME,
			frame);
		if(!result.success)
		{
			error = true;
			printf("Albedo buffer loading failed, reason: %s\n",
					result.error_message.c_str());
			continue;
		}

		normals[frame].resize(3 * IMAGE_WIDTH * IMAGE_HEIGHT);
		result = load_image(normals[frame].data(), NORMAL_FILE_NAME, frame);
		if(!result.success)
		{
			error = true;
			printf("Normal buffer loading failed, reason: %s\n",
					result.error_message.c_str());
			continue;
		}

		positions[frame].resize(3 * IMAGE_WIDTH * IMAGE_HEIGHT);
		result = load_image(positions[frame].data(), POSITION_FILE_NAME, frame);
		if(!result.success)
		{
			error = true;
			printf("Position buffer loading failed, reason: %s\n",
					result.error_message.c_str());
			continue;
		}

		noisy_input[frame].resize(3 * IMAGE_WIDTH * IMAGE_HEIGHT);
		result = load_image(noisy_input[frame].data(), NOISY_FILE_NAME, frame);
		if(!result.success)
		{
			error = true;
			printf("Position buffer loading failed, reason: %s\n",
					result.error_message.c_str());
			continue;
		}
	}

	if(error)
	{
		printf("One or more errors occurred during buffer loading\n");
		return 1;
	}

	// Create OpenCL buffers

	const auto CreateDoubleBuffer = [](cl_context c, cl_mem_flags f, size_t s) -> Double_buffer<cl_mem>
	{
		cl_int ret;
		cl_mem b0 = clCreateBuffer(c, f, s, nullptr, &ret);
		assert(ret == CL_SUCCESS);
		cl_mem b1 = clCreateBuffer(c, f, s, nullptr, &ret);
		assert(ret == CL_SUCCESS);
		return Double_buffer<cl_mem>(b0, b1);
	};

	const auto FreeDoubleBuffer = [](Double_buffer<cl_mem> & buffer)
	{
		cl_int ret;
		ret = clReleaseMemObject(*buffer.previous());
		assert(ret == CL_SUCCESS);
		ret = clReleaseMemObject(*buffer.current());
		assert(ret == CL_SUCCESS);
	};

	
	// Albedo buffer (3 * float32) // TODO: compress this
	const size_t albedo_buffer_size = IMAGE_WIDTH * IMAGE_HEIGHT * 3 * sizeof(float);
	cl_mem albedo_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, albedo_buffer_size, nullptr, &ret);
	assert(ret == CL_SUCCESS);

	// (World) normals buffers (3 * float32) in [-1, +1]^3
	// TODO: compress data? half? -> would require different storage than feature buffer
	const size_t normals_buffer_size = IMAGE_WIDTH * IMAGE_HEIGHT * 3 * sizeof(float);
	Double_buffer<cl_mem> normals_buffer = CreateDoubleBuffer(context, CL_MEM_READ_WRITE, normals_buffer_size);

	// World positions buffer (3 * float32)
	// TODO: normalize in [0, 1] (or [-1, +1])
	const size_t positions_buffer_size = IMAGE_WIDTH * IMAGE_HEIGHT * 3 * sizeof(float);
	Double_buffer<cl_mem> positions_buffer = CreateDoubleBuffer(context, CL_MEM_READ_WRITE, positions_buffer_size);
    
	// Noisy 1spp color buffer (3 * float32)
	const size_t noisy_1spp_buffer_size = IMAGE_WIDTH * IMAGE_HEIGHT * 3 * sizeof(float);
	Double_buffer<cl_mem> noisy_1spp_buffer = CreateDoubleBuffer(context, CL_MEM_READ_WRITE, noisy_1spp_buffer_size);

	// Features buffer (half or single-precision) (3 * float16 or 3 * float32)
	const size_t features_buffer_datatype_size = USE_HALF_PRECISION_IN_FEATURES_DATA ? sizeof(short) : sizeof(float);
	const size_t features_buffer_size = WORKSET_WITH_MARGINS_WIDTH * WORKSET_WITH_MARGINS_HEIGHT * buffer_count * features_buffer_datatype_size;
	cl_mem features_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, features_buffer_size, nullptr, &ret);
	assert(ret == CL_SUCCESS);

	// Noise-free color estimate (3 * float32)
	const size_t noisefree_1spp_size = IMAGE_WIDTH * IMAGE_HEIGHT * 3 * sizeof(float);
	cl_mem noisefree_1spp = clCreateBuffer(context, CL_MEM_READ_WRITE, noisefree_1spp_size, nullptr, &ret);
	assert(ret == CL_SUCCESS);

	// Noise-free accumulated color estimate (3 * float32)
	const size_t noisefree_1spp_accumulated_size = IMAGE_WIDTH * IMAGE_HEIGHT * 3 * sizeof(float);
	Double_buffer<cl_mem> noisefree_1spp_accumulated = CreateDoubleBuffer(context, CL_MEM_READ_WRITE, noisefree_1spp_accumulated_size);

	// Final antialiased color buffer (3 * float32)
	const size_t result_buffer_size = IMAGE_WIDTH * IMAGE_HEIGHT * 3 * sizeof(float);
	Double_buffer<cl_mem> result_buffer = CreateDoubleBuffer(context, CL_MEM_READ_WRITE, result_buffer_size);

	// Previous frame pixel coordinates (after reprojection) (2 * float32)
	const size_t prev_frame_pixel_coords_buffer_size = IMAGE_WIDTH * IMAGE_HEIGHT * 2 * sizeof(float);
	cl_mem prev_frame_pixel_coords_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, prev_frame_pixel_coords_buffer_size, nullptr, &ret);
	assert(ret == CL_SUCCESS);

	// Validity mask of reprojected bilinear samples into previous frame (uchar 8bits)
	const size_t prev_frame_bilinear_samples_validity_mask_size = IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(unsigned char);
	cl_mem prev_frame_bilinear_samples_validity_mask = clCreateBuffer(context, CL_MEM_READ_WRITE, prev_frame_bilinear_samples_validity_mask_size, nullptr, &ret);
	assert(ret == CL_SUCCESS);

	// Tonemapped noise-free color estimate w/ albedo (3 * float32)
	const size_t noisefree_1spp_acc_tonemapped_size = IMAGE_WIDTH * IMAGE_HEIGHT * 3 * sizeof(float);
	cl_mem noisefree_1spp_acc_tonemapped = clCreateBuffer(context, CL_MEM_READ_WRITE, noisefree_1spp_acc_tonemapped_size, nullptr, &ret);
	assert(ret == CL_SUCCESS);

	// Features weights per color channel (x3) (computed by the BMFR) (3 * float32)
	//cl_mem features_weights_buffer(context, CL_MEM_READ_WRITE, (FITTER_KERNEL_GLOBAL_RANGE / 256) * (buffer_count - 3) * 3 * sizeof(cl_float));
	const size_t features_weights_buffer_size = WORKSET_WITH_MARGIN_BLOCK_COUNT * (buffer_count - 3) * 3 * sizeof(float);
	cl_mem features_weights_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, features_weights_buffer_size, nullptr, &ret);
	assert(ret == CL_SUCCESS);

	// Min and max of features values per block (world_positions) (6 * 2 * float32)
	//cl_mem features_min_max_buffer(context, CL_MEM_READ_WRITE, (FITTER_KERNEL_GLOBAL_RANGE / 256) * 6 * sizeof(cl_float2));
	const size_t features_min_max_buffer_size = WORKSET_WITH_MARGIN_BLOCK_COUNT * 6 * 2 * sizeof(float);
	cl_mem features_min_max_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, features_min_max_buffer_size, nullptr, &ret);
	assert(ret == CL_SUCCESS);

	// Number of samples accumulated (for cumulative moving average) (char 8bits)
	const size_t spp_buffer_size = IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(char);
	Double_buffer<cl_mem> spp_buffer = CreateDoubleBuffer(context, CL_MEM_READ_WRITE, spp_buffer_size);

	std::vector<Double_buffer<cl_mem> *> all_double_buffers =
	{
		&normals_buffer,
		&positions_buffer,
		&noisy_1spp_buffer,
		&noisefree_1spp_accumulated,
		&result_buffer,
		&spp_buffer
	};


		// Set kernel arguments
	int arg_index = 0;
	ret = clSetKernelArg(accum_noisy_kernel, arg_index++, sizeof(cl_mem), (void *)&prev_frame_pixel_coords_buffer); // [out] Previous frame pixel coordinates (after reprojection)
	assert(ret == CL_SUCCESS);
	ret = clSetKernelArg(accum_noisy_kernel, arg_index++, sizeof(cl_mem), (void *)&prev_frame_bilinear_samples_validity_mask);	// [out] Validity mask of bilinear samples in previous frame (after reprojection) (i.e valid reprojection = no disoclusion or outside frame)
	assert(ret == CL_SUCCESS);

	arg_index = 0;
	
	#if COMPRESSED_R
	// TODO: replace 'buffer_count-2'
	// Explanations: The number of features M is buffer_count - 3 because buffer_count comprises the 3 noisy 1spp color channels inputs.
	// To this we add 1 because we concatenate z(c) which is the c channel of the noisy path-traced input with makes a size of 'buffer_count-2'.
	// "The Householder QR factorization yields an (M + 1)x(M + 1) upper triangular matrix R(c)"
	// See section 3.3 and 3.4.

	// Computed via sum of arithmetic sequence (that for the upper right triangle):
	//    0  1  2  3  4  5 x
	// 0 00 01 02 03 04 05
	// 1  - 11 12 13 14 15
	// 2  -  - 22 23 24 25
	// 3  -  -  - 33 34 35
	// 4  -  -  -  - 44 45
	// 5  -  -  -  -  - 55
	// y
	// (1 + 2 + ... + buffer_count - 2) = (buffer_count-2+1)*(buffer_count-2)/2 = (buffer_count-1)*(buffer_count-2)/2
	const auto r_size = ((buffer_count - 2) * (buffer_count - 1) / 2) * sizeof(cl_float3);
	#else
	const auto r_size = (buffer_count - 2) * (buffer_count - 2) * sizeof(cl_float3);
	#endif

	// https://www.khronos.org/registry/OpenCL/sdk/1.1/docs/man/xhtml/clSetKernelArg.html
	// Note: For arguments declared with the __local qualifier, the size specified will be the size in bytes of the buffer that must be allocated for the __local argument.

	ret = clSetKernelArg(fitter_kernel, arg_index++, LOCAL_SIZE * sizeof(float), nullptr);		// [local] Size of the shared memory used to perform parrallel reduction (max, min, sum)
	assert(ret == CL_SUCCESS);
	ret = clSetKernelArg(fitter_kernel, arg_index++, BLOCK_PIXELS * sizeof(float), nullptr);	// [local] Shared memory used to store the 'u' vectors
	assert(ret == CL_SUCCESS);
	ret = clSetKernelArg(fitter_kernel, arg_index++, r_size, nullptr);							// [local] Shared memory used to store the R matrix of the QR factorization
	assert(ret == CL_SUCCESS);
	ret = clSetKernelArg(fitter_kernel, arg_index++, sizeof(cl_mem), &features_weights_buffer);					// [out]   Features weights
	assert(ret == CL_SUCCESS);
	ret = clSetKernelArg(fitter_kernel, arg_index++, sizeof(cl_mem), &features_min_max_buffer);					// [out]   Min and max of features values per block (world_positions)
	assert(ret == CL_SUCCESS);

	arg_index = 0;
	ret = clSetKernelArg(weighted_sum_kernel, arg_index++, sizeof(cl_mem), &features_weights_buffer);	// [in]	 Features weights computed by the fitter kernel
	assert(ret == CL_SUCCESS);
	ret = clSetKernelArg(weighted_sum_kernel, arg_index++, sizeof(cl_mem), &features_min_max_buffer);	// [in]  Min and max of features values per block (world_positions)
	assert(ret == CL_SUCCESS);
	ret = clSetKernelArg(weighted_sum_kernel, arg_index++, sizeof(cl_mem), &noisefree_1spp);	// [out] Noise-free color estimate
	assert(ret == CL_SUCCESS);

	arg_index = 0;
	ret = clSetKernelArg(accum_filtered_kernel, arg_index++, sizeof(cl_mem), &noisefree_1spp);					// [in]  Noise free color estimate (computed as the weighted sum of the features)
	assert(ret == CL_SUCCESS);
	ret = clSetKernelArg(accum_filtered_kernel, arg_index++, sizeof(cl_mem), &prev_frame_pixel_coords_buffer);				// [in]  Previous frame pixel coordinates (after reprojection)
	assert(ret == CL_SUCCESS);
	ret = clSetKernelArg(accum_filtered_kernel, arg_index++, sizeof(cl_mem), &prev_frame_bilinear_samples_validity_mask);	// [in]  Validity mask of bilinear samples in previous frame (after reprojection)
	assert(ret == CL_SUCCESS);
	ret = clSetKernelArg(accum_filtered_kernel, arg_index++, sizeof(cl_mem), &albedo_buffer);								// [in]  Albedo buffer of the current frame (non-noisy)
	assert(ret == CL_SUCCESS);
	ret = clSetKernelArg(accum_filtered_kernel, arg_index++, sizeof(cl_mem), &noisefree_1spp_acc_tonemapped);							// [out] Accumulated and tonemapped noise-free color estimate
	assert(ret == CL_SUCCESS);

	arg_index = 0;
	ret = clSetKernelArg(taa_kernel, arg_index++, sizeof(cl_mem), &prev_frame_pixel_coords_buffer);	// [in] Previous frame pixel coordinates (after reprojection)
	assert(ret == CL_SUCCESS);
	ret = clSetKernelArg(taa_kernel, arg_index++, sizeof(cl_mem), &noisefree_1spp_acc_tonemapped);				// [in]	Current frame color buffer
	assert(ret == CL_SUCCESS);

	printf("Run and profile kernels.\n");

	// Note: enqueueNDRangeKernel takes in the global_size and the local_size.
	// In CUDA, a dispatch takes in the grid_size and the block_size.
	// The correspondance is as follow:
	//	global_size = grid_size * block_size
	//	local_size = block_size

	size_t k_workset_with_margin_global_size[] = { WORKSET_WITH_MARGINS_WIDTH, WORKSET_WITH_MARGINS_HEIGHT };
	size_t k_workset_global_size[] = { WORKSET_WIDTH, WORKSET_HEIGHT };
	size_t k_local_size[] = { LOCAL_WIDTH, LOCAL_HEIGHT };
	size_t k_fitter_global_size[] = { FITTER_KERNEL_GLOBAL_RANGE };
	size_t k_fitter_local_size[] = { LOCAL_SIZE };

	// Note: in real use case there would not be WriteBuffer and ReadBuffer function calls
	// because the input data comes from the path tracer and output goes to the screen
	for(int frame = 0; frame < FRAME_COUNT; ++frame)
	{
		const cl_bool blocking_write = true;
		// https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/clEnqueueWriteBuffer.html
		// Enqueue commands to write to a buffer object from host memory (= cudaMemcpy(..., cudaMemcpyHostToDevice))
		ret = clEnqueueWriteBuffer(command_queue, albedo_buffer, blocking_write, 0, albedo_buffer_size, albedos[frame].data(), 0, nullptr, nullptr);
		assert(ret == CL_SUCCESS);
		ret = clEnqueueWriteBuffer(command_queue, *normals_buffer.current(), blocking_write, 0, normals_buffer_size, normals[frame].data(), 0, nullptr, nullptr);
		assert(ret == CL_SUCCESS);
		ret = clEnqueueWriteBuffer(command_queue, *positions_buffer.current(), blocking_write, 0, positions_buffer_size, positions[frame].data(), 0, nullptr, nullptr);
		assert(ret == CL_SUCCESS);
		ret = clEnqueueWriteBuffer(command_queue, *noisy_1spp_buffer.current(), blocking_write, 0, noisy_1spp_buffer_size, noisy_input[frame].data(), 0, nullptr, nullptr);
		assert(ret == CL_SUCCESS);

		// Phase I:
		//  - accumulate noisy 1spp
		//  - compute previous frame pixel coordinates (after reprojection)
		//  - generate validity bit mask of bilinear samples of previous frame
		//  - concatenate the different features in a single buffer
		// Note: On the first frame accum_noisy_kernel just copies to the features_buffer
		arg_index = 2;
		ret = clSetKernelArg(accum_noisy_kernel, arg_index++, sizeof(cl_mem), normals_buffer.current());			// [in]  Current  (world) normals
		assert(ret == CL_SUCCESS);
		ret = clSetKernelArg(accum_noisy_kernel, arg_index++, sizeof(cl_mem), normals_buffer.previous());			// [in]  Previous (world) normals
		assert(ret == CL_SUCCESS);
		ret = clSetKernelArg(accum_noisy_kernel, arg_index++, sizeof(cl_mem), positions_buffer.current());		// [in]  Current  world positions
		assert(ret == CL_SUCCESS);
		ret = clSetKernelArg(accum_noisy_kernel, arg_index++, sizeof(cl_mem), positions_buffer.previous());		// [in]  Previous world positions
		assert(ret == CL_SUCCESS);
		ret = clSetKernelArg(accum_noisy_kernel, arg_index++, sizeof(cl_mem), noisy_1spp_buffer.current());	// [out] Current  noisy 1spp color
		assert(ret == CL_SUCCESS);
		ret = clSetKernelArg(accum_noisy_kernel, arg_index++, sizeof(cl_mem), noisy_1spp_buffer.previous());// [in]  Previous noisy 1spp color
		assert(ret == CL_SUCCESS);
		ret = clSetKernelArg(accum_noisy_kernel, arg_index++, sizeof(cl_mem), spp_buffer.previous());				// [in]  Previous number of samples accumulated (for CMA)
		assert(ret == CL_SUCCESS);
		ret = clSetKernelArg(accum_noisy_kernel, arg_index++, sizeof(cl_mem), spp_buffer.current());				// [out] Current  number of samples accumulated (for CMA)
		assert(ret == CL_SUCCESS);
		ret = clSetKernelArg(accum_noisy_kernel, arg_index++, sizeof(cl_mem), &features_buffer);					// [out] Features buffer (half or single-precision)
		assert(ret == CL_SUCCESS);
		const int matrix_index = frame == 0 ? 0 : frame - 1;
		ret = clSetKernelArg(accum_noisy_kernel, arg_index++, sizeof(cl_float16), &(camera_matrices[matrix_index][0][0])); // [in] ViewProj matrix of previous frame
		assert(ret == CL_SUCCESS);
		ret = clSetKernelArg(accum_noisy_kernel, arg_index++, sizeof(cl_float2), &(pixel_offsets[frame][0]));
		assert(ret == CL_SUCCESS);
		ret = clSetKernelArg(accum_noisy_kernel, arg_index++, sizeof(cl_int), &frame); // [in] Current frame number
		assert(ret == CL_SUCCESS);
		ret = clEnqueueNDRangeKernel(command_queue, accum_noisy_kernel, 2, NULL, k_workset_with_margin_global_size, k_local_size, 0, NULL, NULL);
		assert(ret == CL_SUCCESS);

		if(0)
		{
			std::vector<float> debugData;
			debugData.resize(features_buffer_size / sizeof(float));

			for(auto i = 0; i < debugData.size(); ++i)
			{
				debugData[i] = 0.42f;
			}

			ret = clEnqueueWriteBuffer(command_queue, features_buffer, blocking_write, 0, features_buffer_size, debugData.data(), 0, nullptr, nullptr);
			assert(ret == CL_SUCCESS);

			ret = clFlush(command_queue);
			assert(ret == CL_SUCCESS);
			ret = clFinish(command_queue);
			assert(ret == CL_SUCCESS);
		}
 
		// Phase II: Blockwise Multi-Order Feature Regression (BMFR)
		// -> compute features weightss
		arg_index = 5;
		ret = clSetKernelArg(fitter_kernel, arg_index++, sizeof(cl_mem), &features_buffer);			// [in] Features buffer (half or single-precision)
		assert(ret == CL_SUCCESS);
		ret = clSetKernelArg(fitter_kernel, arg_index++, sizeof(cl_int), &frame);  // [in] Current frame number
		assert(ret == CL_SUCCESS);
		ret = clEnqueueNDRangeKernel(command_queue, fitter_kernel, 1, NULL, k_fitter_global_size, k_fitter_local_size, 0, NULL, NULL);
		assert(ret == CL_SUCCESS);

		// Phase II: Compute noise free color estimate (weighted sum of features)
		arg_index = 3;
		ret = clSetKernelArg(weighted_sum_kernel, arg_index++, sizeof(cl_mem), normals_buffer.current());			 // [in] Current (world) normals
		assert(ret == CL_SUCCESS);
		ret = clSetKernelArg(weighted_sum_kernel, arg_index++, sizeof(cl_mem), positions_buffer.current());		 // [in] Current world positions
		assert(ret == CL_SUCCESS);
		ret = clSetKernelArg(weighted_sum_kernel, arg_index++, sizeof(cl_mem), noisy_1spp_buffer.current()); // [in] Current noisy 1spp color (only used for debugging)
		assert(ret == CL_SUCCESS);
		ret = clSetKernelArg(weighted_sum_kernel, arg_index++, sizeof(cl_int), &frame);			 // [in] Current frame number
		assert(ret == CL_SUCCESS);
		ret = clEnqueueNDRangeKernel(command_queue, weighted_sum_kernel, 2, NULL, k_workset_global_size, k_local_size, 0, NULL, NULL);
		assert(ret == CL_SUCCESS);

		// Phase III: Postprocessing
		// -> accumulate noise-free color estimate + output a tonemapped version
		arg_index = 5;
		ret = clSetKernelArg(accum_filtered_kernel, arg_index++, sizeof(cl_mem), spp_buffer.current());	// [in]	 Current number of samples accumulated (for CMA)
		assert(ret == CL_SUCCESS);
		ret = clSetKernelArg(accum_filtered_kernel, arg_index++, sizeof(cl_mem), noisefree_1spp_accumulated.previous()); // [in]  Previous frame noise-free accumulated color estimate 
		assert(ret == CL_SUCCESS);
		ret = clSetKernelArg(accum_filtered_kernel, arg_index++, sizeof(cl_mem), noisefree_1spp_accumulated.current());	 // [out] Current frame noise-free accumulated color estimate
		assert(ret == CL_SUCCESS);
		ret = clSetKernelArg(accum_filtered_kernel, arg_index++, sizeof(cl_int), &frame);	// [in]  Current frame number
		assert(ret == CL_SUCCESS);
		ret = clEnqueueNDRangeKernel(command_queue, accum_filtered_kernel, 2, NULL, k_workset_global_size, k_local_size, 0, NULL, NULL);
		assert(ret == CL_SUCCESS);

		// Phase III: Temporal antialiasing
		arg_index = 2;
		ret = clSetKernelArg(taa_kernel, arg_index++, sizeof(cl_mem), result_buffer.current());	// [out] Antialiased frame color buffer
		assert(ret == CL_SUCCESS);
		ret = clSetKernelArg(taa_kernel, arg_index++, sizeof(cl_mem), result_buffer.previous());	// [in]  Previous frame color buffer
		assert(ret == CL_SUCCESS);
		ret = clSetKernelArg(taa_kernel, arg_index++, sizeof(cl_int), &frame);	// [in]  Current frame number
		assert(ret == CL_SUCCESS);
		ret = clEnqueueNDRangeKernel(command_queue, taa_kernel, 2, NULL, k_workset_global_size, k_local_size, 0, NULL, NULL);
		assert(ret == CL_SUCCESS);

		#if ENABLE_DEBUG_OUTPUT_TMP_DATA
		if(frame == DEBUG_OUTPUT_FRAME_NUMBER)
		{
			tmpData.normals.resize(normals_buffer_size / sizeof(float));
			ret = clEnqueueReadBuffer(command_queue, *normals_buffer.current(), false, 0, normals_buffer_size, tmpData.normals.data(), 0, NULL, NULL);

			tmpData.positions.resize(positions_buffer_size / sizeof(float));
			ret = clEnqueueReadBuffer(command_queue, *positions_buffer.current(), false, 0, positions_buffer_size, tmpData.positions.data(), 0, NULL, NULL);

			tmpData.noisy_1spp.resize(noisy_1spp_buffer_size / sizeof(float));
			ret = clEnqueueReadBuffer(command_queue, *noisy_1spp_buffer.current(), false, 0, noisy_1spp_buffer_size, tmpData.noisy_1spp.data(), 0, NULL, NULL);
			assert(ret == CL_SUCCESS);

			tmpData.prev_frame_pixel_coords_buffer.resize(prev_frame_pixel_coords_buffer_size / sizeof(float));
			ret = clEnqueueReadBuffer(command_queue, prev_frame_pixel_coords_buffer, false, 0, prev_frame_pixel_coords_buffer_size, tmpData.prev_frame_pixel_coords_buffer.data(), 0, NULL, NULL);
			assert(ret == CL_SUCCESS);

			tmpData.prev_frame_bilinear_samples_validity_mask.resize(prev_frame_bilinear_samples_validity_mask_size / sizeof(unsigned char));
			ret = clEnqueueReadBuffer(command_queue, prev_frame_bilinear_samples_validity_mask, false, 0, prev_frame_bilinear_samples_validity_mask_size, tmpData.prev_frame_bilinear_samples_validity_mask.data(), 0, NULL, NULL);
			assert(ret == CL_SUCCESS);

			tmpData.spp.resize(spp_buffer_size / sizeof(unsigned char));
			ret = clEnqueueReadBuffer(command_queue, *spp_buffer.current(), false, 0, spp_buffer_size, tmpData.spp.data(), 0, NULL, NULL);
			assert(ret == CL_SUCCESS);

			tmpData.features_buffer.resize(features_buffer_size / sizeof(float));
			ret = clEnqueueReadBuffer(command_queue, features_buffer, false, 0, features_buffer_size, tmpData.features_buffer.data(), 0, NULL, NULL);
			assert(ret == CL_SUCCESS);

			tmpData.features_weights_buffer.resize(features_weights_buffer_size / sizeof(float));
			ret = clEnqueueReadBuffer(command_queue, features_weights_buffer, false, 0, features_weights_buffer_size, tmpData.features_weights_buffer.data(), 0, NULL, NULL);
			assert(ret == CL_SUCCESS);

			tmpData.features_min_max_buffer.resize(features_min_max_buffer_size / sizeof(float));
			ret = clEnqueueReadBuffer(command_queue, features_min_max_buffer, false, 0, features_min_max_buffer_size, tmpData.features_min_max_buffer.data(), 0, NULL, NULL);
			assert(ret == CL_SUCCESS);

			tmpData.noisefree_1spp.resize(noisefree_1spp_size / sizeof(float));
			ret = clEnqueueReadBuffer(command_queue, noisefree_1spp, false, 0, noisefree_1spp_size, tmpData.noisefree_1spp.data(), 0, NULL, NULL);
			assert(ret == CL_SUCCESS);

			tmpData.noisefree_1spp_accumulated.resize(noisefree_1spp_accumulated_size / sizeof(float));
			ret = clEnqueueReadBuffer(command_queue, *noisefree_1spp_accumulated.current(), false, 0, noisefree_1spp_accumulated_size, tmpData.noisefree_1spp_accumulated.data(), 0, NULL, NULL);
			assert(ret == CL_SUCCESS);

			tmpData.noisefree_1spp_acc_tonemapped.resize(noisefree_1spp_acc_tonemapped_size / sizeof(float));
			ret = clEnqueueReadBuffer(command_queue, noisefree_1spp_acc_tonemapped, false, 0, noisefree_1spp_acc_tonemapped_size, tmpData.noisefree_1spp_acc_tonemapped.data(), 0, NULL, NULL);
			assert(ret == CL_SUCCESS);

			tmpData.result.resize(result_buffer_size / sizeof(float));
			ret = clEnqueueReadBuffer(command_queue, *result_buffer.current(), false, 0, result_buffer_size, tmpData.result.data(), 0, NULL, NULL);
			assert(ret == CL_SUCCESS);

			ret = clFlush(command_queue);
			assert(ret == CL_SUCCESS);
			ret = clFinish(command_queue);
			assert(ret == CL_SUCCESS);
			
			return 0;
		}
		#endif

		// This is not timed because in real use case the result is stored to frame buffer
		ret = clEnqueueReadBuffer(command_queue, *result_buffer.current(), false, 0, OUTPUT_SIZE * 3 * sizeof(cl_float), out_data[frame].data(), 0, NULL, NULL);
		assert(ret == CL_SUCCESS);

		// Swap all double buffers
		std::for_each(all_double_buffers.begin(), all_double_buffers.end(), std::bind(&Double_buffer<cl_mem>::swap, std::placeholders::_1));
	}
    
	// Clean up
	ret = clFlush(command_queue);
	assert(ret == CL_SUCCESS);
	ret = clFinish(command_queue);
	assert(ret == CL_SUCCESS);
	ret = clReleaseKernel(accum_noisy_kernel);
	assert(ret == CL_SUCCESS);
	ret = clReleaseKernel(fitter_kernel);
	assert(ret == CL_SUCCESS);
	ret = clReleaseKernel(weighted_sum_kernel);
	assert(ret == CL_SUCCESS);
	ret = clReleaseKernel(accum_filtered_kernel);
	assert(ret == CL_SUCCESS);
	ret = clReleaseKernel(taa_kernel);
	assert(ret == CL_SUCCESS);
	ret = clReleaseProgram(program);
	assert(ret == CL_SUCCESS);
	FreeDoubleBuffer(normals_buffer);
	FreeDoubleBuffer(positions_buffer);
	FreeDoubleBuffer(noisy_1spp_buffer);
	ret = clReleaseMemObject(features_buffer);
	assert(ret == CL_SUCCESS);
	ret = clReleaseMemObject(noisefree_1spp);
	assert(ret == CL_SUCCESS);
	FreeDoubleBuffer(noisefree_1spp_accumulated);
	FreeDoubleBuffer(result_buffer);
	ret = clReleaseMemObject(prev_frame_pixel_coords_buffer);
	assert(ret == CL_SUCCESS);
	ret = clReleaseMemObject(prev_frame_bilinear_samples_validity_mask);
	assert(ret == CL_SUCCESS);
	ret = clReleaseMemObject(albedo_buffer);
	assert(ret == CL_SUCCESS);
	ret = clReleaseMemObject(noisefree_1spp_acc_tonemapped);
	assert(ret == CL_SUCCESS);
	ret = clReleaseMemObject(features_weights_buffer);
	assert(ret == CL_SUCCESS);
	ret = clReleaseMemObject(features_min_max_buffer);
	assert(ret == CL_SUCCESS);
	FreeDoubleBuffer(spp_buffer);
	ret = clReleaseCommandQueue(command_queue);
	assert(ret == CL_SUCCESS);
	ret = clReleaseContext(context);
	assert(ret == CL_SUCCESS);

	// Store results
	error = false;
	#pragma omp parallel for
	for(int frame = 0; frame < FRAME_COUNT; ++frame)
	{
		if(error)
			continue;

		// Output image
		std::string output_file_name = "outputs/result_" + std::to_string(frame) + "_opencl.png";
		// Crops back from WORKSET_SIZE to IMAGE_SIZE
		OpenImageIO::ImageSpec spec(IMAGE_WIDTH, IMAGE_HEIGHT, 3,
									OpenImageIO::TypeDesc::FLOAT);
		std::unique_ptr<OpenImageIO::ImageOutput>
			out(OpenImageIO::ImageOutput::create(output_file_name));
		if(out && out->open(output_file_name, spec))
		{
			out->write_image(OpenImageIO::TypeDesc::FLOAT, out_data[frame].data(),
								3 * sizeof(cl_float), WORKSET_WIDTH * 3 * sizeof(cl_float), 0);
			out->close();
		}
		else
		{
			printf("Can't create image file on disk to location %s\n",
					output_file_name.c_str());
			error = true;
			continue;
		}
	}

	if(error)
	{
		printf("One or more errors occurred during image saving\n");
		return 1;
	}

	return 0;
}
