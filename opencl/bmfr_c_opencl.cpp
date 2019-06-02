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

#include "bmfr_c_opencl.hpp"
#include "OpenImageIO/imageio.h"
#include "CLUtils/CLUtils.hpp"
#include <functional>
#include <memory>


// ### Choose your OpenCL device and platform with these defines ###
#define PLATFORM_INDEX 1
#define DEVICE_INDEX 0


// ### Edit these defines if you have different input ###

#define KERNEL_FILENAME "bmfr.cl"

// TODO: turn size and dependent constants into variables (that will be baked as constant inside the kernel)

// TODO detect IMAGE_SIZES automatically from the input files
#define IMAGE_WIDTH 1280
#define IMAGE_HEIGHT 720

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
#define COMPRESSED_R 1

// If 1 stores tmp_data to private memory when it is loaded for dot product calculation
#define CACHE_TMP_DATA 1

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

static Operation_result read_image_file(const std::string &file_name, const int frame, float *buffer)
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

static Operation_result load_image(cl_float *image, const std::string file_name, const int frame)
{
    Operation_result result = read_image_file(file_name, frame, image);
    if(!result.success)
        return result;

    return {true};
}

#define K_OPENCL_CHECK(openclFunc) \
	do { \
		/*printf(#cudaFunc "\n");\*/ \
		cl_int ret = openclFunc; \
		if(ret != CL_SUCCESS) \
		{ \
			printf("OpenCL error: %d\n", ret); \
			__debugbreak(); \
		} \
	} while(0)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
 
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
 
#define MAX_SOURCE_SIZE (0x100000)

// Only work with 3 channels float32 image
static void SaveDevice3Float32ImageToDisk(
	std::string const & filename,
	int frame,
	cl_command_queue const & command_queue,
	cl_mem const & buffer,
	BufferDesc const & desc
)
{
	const size_t datasize = desc.byte_size;
	const size_t numelem  = datasize / sizeof(float);
	std::vector<float> outdata;
	outdata.resize(numelem);

	K_OPENCL_CHECK(clEnqueueReadBuffer(command_queue, buffer, false, 0, datasize, outdata.data(), 0, NULL, NULL));
	K_OPENCL_CHECK(clFlush(command_queue));
	K_OPENCL_CHECK(clFinish(command_queue));

	std::string output_filename = OUTPUT_FOLDER + filename + "_" + std::to_string(frame) + "_opencl.png";

	// Output image
	printf("  Save image %s\n", output_filename.c_str());

    OpenImageIO::ImageSpec spec(desc.w, desc.h, 3, OpenImageIO::TypeDesc::FLOAT);
    std::unique_ptr<OpenImageIO::ImageOutput> out(OpenImageIO::ImageOutput::create(output_filename));
    if(out && out->open(output_filename, spec))
    {
        out->write_image(OpenImageIO::TypeDesc::FLOAT, outdata.data(), desc.x_stride, desc.y_stride, 0);
        out->close();
    }
    else
    {
        printf("  Can't create image file on disk to location %s\n", output_filename.c_str());
    }
}

int bmfr_c_opencl(TmpData & tmpData)
{
    printf("Initialize.\n");

	// Based on: https://gist.github.com/courtneyfaulkner/7919509
	{
		cl_uint i, j;
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
	K_OPENCL_CHECK(clGetPlatformIDs(0, NULL, &ret_num_platforms));
	cl_platform_id *platforms = NULL;
	platforms = (cl_platform_id*)malloc(ret_num_platforms*sizeof(cl_platform_id));
	K_OPENCL_CHECK(clGetPlatformIDs(ret_num_platforms, platforms, NULL));
	K_OPENCL_CHECK(clGetDeviceIDs(platforms[1], CL_DEVICE_TYPE_ALL, 1, &device_id, &ret_num_devices));

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
    K_OPENCL_CHECK(clGetPlatformIDs(1, &platform_id, &ret_num_platforms));
	K_OPENCL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices));

	// Print device name
    size_t valueSize;
    clGetDeviceInfo(device_id, CL_DEVICE_NAME, 0, NULL, &valueSize);
    char * value = (char*) malloc(valueSize);
    clGetDeviceInfo(device_id, CL_DEVICE_NAME, valueSize, value, NULL);
    printf("Selected device: %s\n", value);
    free(value);
#endif

	cl_int ret = CL_SUCCESS;

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
	if(clBuildProgram(program, 1, &device_id, build_options.str().c_str(), NULL, NULL) != CL_SUCCESS)
	{
		size_t len = 0;
		K_OPENCL_CHECK(clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &len));
		char *buffer = (char*)malloc(len * sizeof(char));
		K_OPENCL_CHECK(clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, len, buffer, NULL));

		printf("Compilation failed: %s\n", buffer);
	}

	// Phase I
	// 3.2 Preprocessing: temporal accumulation of the noisy 1 spp data, which reprojects the previous accumulated data to the new camera frame
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

	struct FrameInputData
	{
		std::vector<float> albedos;
		std::vector<float> normals;
		std::vector<float> positions;
		std::vector<float> noisy1spps;
	} frameInput;

	std::vector<float> result;
    result.resize(3 * OUTPUT_SIZE);

	// Allocate frame input data buffers
    frameInput.albedos.resize(3 * IMAGE_WIDTH * IMAGE_HEIGHT);
    frameInput.normals.resize(3 * IMAGE_WIDTH * IMAGE_HEIGHT);
    frameInput.positions.resize(3 * IMAGE_WIDTH * IMAGE_HEIGHT);
    frameInput.noisy1spps.resize(3 * IMAGE_WIDTH * IMAGE_HEIGHT);


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
		K_OPENCL_CHECK(clReleaseMemObject(*buffer.previous()));
		K_OPENCL_CHECK(clReleaseMemObject(*buffer.current()));
	};

	const auto LoadFrameInputData = [](FrameInputData & frameInputData, int frame) -> bool
	{
		printf("  Loading data of frame %d\n", frame);

        Operation_result result = load_image(frameInputData.albedos.data(), ALBEDO_FILE_NAME, frame);
        if(!result.success)
        {
            printf("Albedo buffer loading failed, reason: %s (frame %d)\n", result.error_message.c_str(), frame);
            return false;
        }

        result = load_image(frameInputData.normals.data(), NORMAL_FILE_NAME, frame);
        if(!result.success)
        {
            printf("Normal buffer loading failed, reason: %s (frame %d)\n", result.error_message.c_str(), frame);
            return false;
        }

        result = load_image(frameInputData.positions.data(), POSITION_FILE_NAME, frame);
        if(!result.success)
        {
            printf("Position buffer loading failed, reason: %s (frame %d)\n", result.error_message.c_str(), frame);
            return false;
        }

        result = load_image(frameInputData.noisy1spps.data(), NOISY_FILE_NAME, frame);
        if(!result.success)
        {
            printf("Position buffer loading failed, reason: %s (frame %d)\n", result.error_message.c_str(), frame);
            return false;
        }
            
		return true;
	};

	// Create OpenCL buffers

	printf("\nAllocate OpenCL buffers\n");

	const size_t w = IMAGE_WIDTH;
	const size_t h = IMAGE_HEIGHT;
	
	// Albedo buffer (3 * float32) // TODO: compress this
	BufferDesc albedoBufferDesc = GetAlbedoBufferDesc(w, h);
	const size_t albedo_buffer_size = albedoBufferDesc.byte_size;
	cl_mem albedo_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, albedo_buffer_size, nullptr, &ret);
	assert(ret == CL_SUCCESS);

	// (World) normals buffers (3 * float32) in [-1, +1]^3
	// TODO: compress data? half? -> would require different storage than feature buffer
	BufferDesc normalsBufferDesc = GetNormalsBufferDesc(w, h);
	const size_t normals_buffer_size = normalsBufferDesc.byte_size;
	Double_buffer<cl_mem> normals_buffer = CreateDoubleBuffer(context, CL_MEM_READ_WRITE, normals_buffer_size);

	// World positions buffer (3 * float32)
	// TODO: normalize in [0, 1] (or [-1, +1])
	BufferDesc positionsBufferDesc = GetPositionsBufferDesc(w, h);
	const size_t positions_buffer_size = positionsBufferDesc.byte_size;
	Double_buffer<cl_mem> positions_buffer = CreateDoubleBuffer(context, CL_MEM_READ_WRITE, positions_buffer_size);
    
	// Noisy 1spp color buffer (3 * float32)
	BufferDesc noisy1sppBufferDesc = GetNoisy1sppBufferDesc(w, h);
	const size_t noisy_1spp_buffer_size = noisy1sppBufferDesc.byte_size;
	Double_buffer<cl_mem> noisy_1spp_buffer = CreateDoubleBuffer(context, CL_MEM_READ_WRITE, noisy_1spp_buffer_size);

	// Features buffer (half or single-precision) (3 * float16 or 3 * float32)
	const size_t features_buffer_datatype_size = USE_HALF_PRECISION_IN_FEATURES_DATA ? sizeof(short) : sizeof(float);
	const size_t features_buffer_size = WORKSET_WITH_MARGINS_WIDTH * WORKSET_WITH_MARGINS_HEIGHT * buffer_count * features_buffer_datatype_size;
	cl_mem features_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, features_buffer_size, nullptr, &ret);
	assert(ret == CL_SUCCESS);

	// Noise-free color estimate (3 * float32)
	BufferDesc noiseFree1sppBufferDesc = GetNoiseFree1sppBufferDesc(w, h);
	const size_t noisefree_1spp_size = noiseFree1sppBufferDesc.byte_size;
	cl_mem noisefree_1spp = clCreateBuffer(context, CL_MEM_READ_WRITE, noisefree_1spp_size, nullptr, &ret);
	assert(ret == CL_SUCCESS);

	// Noise-free accumulated color estimate (3 * float32)
	const BufferDesc noiseFree1sppAccumulatedBufferDesc = GetNoiseFree1sppAccumulatedBufferDesc(w, h);
	const size_t noisefree_1spp_accumulated_size = noiseFree1sppAccumulatedBufferDesc.byte_size;
	Double_buffer<cl_mem> noisefree_1spp_accumulated = CreateDoubleBuffer(context, CL_MEM_READ_WRITE, noisefree_1spp_accumulated_size);

	// Final antialiased color buffer (3 * float32)
	const BufferDesc resultBufferDesc = GetResultBufferDesc(w, h);
	const size_t result_buffer_size = resultBufferDesc.byte_size;
	Double_buffer<cl_mem> result_buffer = CreateDoubleBuffer(context, CL_MEM_READ_WRITE, result_buffer_size);

	// Previous frame pixel coordinates (after reprojection) (2 * float32)
	const BufferDesc prevFramePixelCoordsBufferDesc = GetPrevFramePixelCoordsBufferDesc(w, h);
	const size_t prev_frame_pixel_coords_buffer_size = prevFramePixelCoordsBufferDesc.byte_size;
	cl_mem prev_frame_pixel_coords_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, prev_frame_pixel_coords_buffer_size, nullptr, &ret);
	assert(ret == CL_SUCCESS);

	// Validity mask of reprojected bilinear samples into previous frame (uchar 8bits)
	const BufferDesc prevFrameBilinearSamplesValidityMaskBufferDesc = GetPrevFrameBilinearSamplesValidityMaskBufferDesc(w, h);
	const size_t prev_frame_bilinear_samples_validity_mask_size = prevFrameBilinearSamplesValidityMaskBufferDesc.byte_size;
	cl_mem prev_frame_bilinear_samples_validity_mask = clCreateBuffer(context, CL_MEM_READ_WRITE, prev_frame_bilinear_samples_validity_mask_size, nullptr, &ret);
	assert(ret == CL_SUCCESS);

	// Tonemapped noise-free color estimate w/ albedo (3 * float32)
	const BufferDesc noiseFree1sppAccTonemappedBufferDesc = GetNoiseFree1sppAccTonemappedBufferDesc(w, h);
	const size_t noisefree_1spp_acc_tonemapped_size = noiseFree1sppAccTonemappedBufferDesc.byte_size;
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
	const BufferDesc sppBufferDesc = GetSppBufferDesc(w, h);
	const size_t spp_buffer_size = sppBufferDesc.byte_size;
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
	K_OPENCL_CHECK(clSetKernelArg(accum_noisy_kernel, arg_index++, sizeof(cl_mem), (void *)&prev_frame_pixel_coords_buffer)); // [out] Previous frame pixel coordinates (after reprojection)
	K_OPENCL_CHECK(clSetKernelArg(accum_noisy_kernel, arg_index++, sizeof(cl_mem), (void *)&prev_frame_bilinear_samples_validity_mask));	// [out] Validity mask of bilinear samples in previous frame (after reprojection) (i.e valid reprojection = no disoclusion or outside frame)

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

	K_OPENCL_CHECK(clSetKernelArg(fitter_kernel, arg_index++, LOCAL_SIZE * sizeof(float), nullptr));		// [local] Size of the shared memory used to perform parrallel reduction (max, min, sum)
	K_OPENCL_CHECK(clSetKernelArg(fitter_kernel, arg_index++, BLOCK_PIXELS * sizeof(float), nullptr));		// [local] Shared memory used to store the 'u' vectors
	K_OPENCL_CHECK(clSetKernelArg(fitter_kernel, arg_index++, r_size, nullptr));							// [local] Shared memory used to store the R matrix of the QR factorization
	K_OPENCL_CHECK(clSetKernelArg(fitter_kernel, arg_index++, sizeof(cl_mem), &features_weights_buffer));	// [out]   Features weights
	K_OPENCL_CHECK(clSetKernelArg(fitter_kernel, arg_index++, sizeof(cl_mem), &features_min_max_buffer));	// [out]   Min and max of features values per block (world_positions)

	arg_index = 0;
	K_OPENCL_CHECK(clSetKernelArg(weighted_sum_kernel, arg_index++, sizeof(cl_mem), &features_weights_buffer));	// [in]	 Features weights computed by the fitter kernel
	K_OPENCL_CHECK(clSetKernelArg(weighted_sum_kernel, arg_index++, sizeof(cl_mem), &features_min_max_buffer));	// [in]  Min and max of features values per block (world_positions)
	K_OPENCL_CHECK(clSetKernelArg(weighted_sum_kernel, arg_index++, sizeof(cl_mem), &noisefree_1spp));			// [out] Noise-free color estimate

	arg_index = 0;
	K_OPENCL_CHECK(clSetKernelArg(accum_filtered_kernel, arg_index++, sizeof(cl_mem), &noisefree_1spp));							// [in]  Noise free color estimate (computed as the weighted sum of the features)
	K_OPENCL_CHECK(clSetKernelArg(accum_filtered_kernel, arg_index++, sizeof(cl_mem), &prev_frame_pixel_coords_buffer));			// [in]  Previous frame pixel coordinates (after reprojection)
	K_OPENCL_CHECK(clSetKernelArg(accum_filtered_kernel, arg_index++, sizeof(cl_mem), &prev_frame_bilinear_samples_validity_mask));	// [in]  Validity mask of bilinear samples in previous frame (after reprojection)
	K_OPENCL_CHECK(clSetKernelArg(accum_filtered_kernel, arg_index++, sizeof(cl_mem), &albedo_buffer));								// [in]  Albedo buffer of the current frame (non-noisy)
	K_OPENCL_CHECK(clSetKernelArg(accum_filtered_kernel, arg_index++, sizeof(cl_mem), &noisefree_1spp_acc_tonemapped));				// [out] Accumulated and tonemapped noise-free color estimate

	arg_index = 0;
	K_OPENCL_CHECK(clSetKernelArg(taa_kernel, arg_index++, sizeof(cl_mem), &prev_frame_pixel_coords_buffer));	// [in] Previous frame pixel coordinates (after reprojection)
	K_OPENCL_CHECK(clSetKernelArg(taa_kernel, arg_index++, sizeof(cl_mem), &noisefree_1spp_acc_tonemapped));	// [in]	Current frame color buffer

	printf("Run kernels.\n");

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
		printf("Frame %d\n", frame);

		printf("  Load frame input data from disk\n");
		LoadFrameInputData(frameInput, frame);

		printf("  Transfert data from host to device\n");
		const cl_bool blocking_write = true;
		K_OPENCL_CHECK(clEnqueueWriteBuffer(command_queue, albedo_buffer, blocking_write, 0, frameInput.albedos.size() * sizeof(float), frameInput.albedos.data(), 0, nullptr, nullptr));
		K_OPENCL_CHECK(clEnqueueWriteBuffer(command_queue, *normals_buffer.current(), blocking_write, 0, frameInput.normals.size() * sizeof(float), frameInput.normals.data(), 0, nullptr, nullptr));
		K_OPENCL_CHECK(clEnqueueWriteBuffer(command_queue, *positions_buffer.current(), blocking_write, 0, frameInput.positions.size() * sizeof(float), frameInput.positions.data(), 0, nullptr, nullptr));
		K_OPENCL_CHECK(clEnqueueWriteBuffer(command_queue, *noisy_1spp_buffer.current(), blocking_write, 0, frameInput.noisy1spps.size() * sizeof(float), frameInput.noisy1spps.data(), 0, nullptr, nullptr));

		// Phase I:
		//  - accumulate noisy 1spp
		//  - compute previous frame pixel coordinates (after reprojection)
		//  - generate validity bit mask of bilinear samples of previous frame
		//  - concatenate the different features in a single buffer
		// Note: On the first frame accum_noisy_kernel just copies to the features_buffer
		arg_index = 2;
		K_OPENCL_CHECK(clSetKernelArg(accum_noisy_kernel, arg_index++, sizeof(cl_mem), normals_buffer.current()));			// [in]  Current  (world) normals
		K_OPENCL_CHECK(clSetKernelArg(accum_noisy_kernel, arg_index++, sizeof(cl_mem), normals_buffer.previous()));			// [in]  Previous (world) normals
		K_OPENCL_CHECK(clSetKernelArg(accum_noisy_kernel, arg_index++, sizeof(cl_mem), positions_buffer.current()));		// [in]  Current  world positions
		K_OPENCL_CHECK(clSetKernelArg(accum_noisy_kernel, arg_index++, sizeof(cl_mem), positions_buffer.previous()));		// [in]  Previous world positions
		K_OPENCL_CHECK(clSetKernelArg(accum_noisy_kernel, arg_index++, sizeof(cl_mem), noisy_1spp_buffer.current()));		// [out] Current  noisy 1spp color
		K_OPENCL_CHECK(clSetKernelArg(accum_noisy_kernel, arg_index++, sizeof(cl_mem), noisy_1spp_buffer.previous()));		// [in]  Previous noisy 1spp color
		K_OPENCL_CHECK(clSetKernelArg(accum_noisy_kernel, arg_index++, sizeof(cl_mem), spp_buffer.previous()));				// [in]  Previous number of samples accumulated (for CMA)
		K_OPENCL_CHECK(clSetKernelArg(accum_noisy_kernel, arg_index++, sizeof(cl_mem), spp_buffer.current()));				// [out] Current  number of samples accumulated (for CMA)
		K_OPENCL_CHECK(clSetKernelArg(accum_noisy_kernel, arg_index++, sizeof(cl_mem), &features_buffer));					// [out] Features buffer (half or single-precision)
		const int matrix_index = frame == 0 ? 0 : frame - 1;
		K_OPENCL_CHECK(clSetKernelArg(accum_noisy_kernel, arg_index++, sizeof(cl_float16), &(camera_matrices[matrix_index][0][0]))); // [in] ViewProj matrix of previous frame
		K_OPENCL_CHECK(clSetKernelArg(accum_noisy_kernel, arg_index++, sizeof(cl_float2), &(pixel_offsets[frame][0])));
		K_OPENCL_CHECK(clSetKernelArg(accum_noisy_kernel, arg_index++, sizeof(cl_int), &frame)); // [in] Current frame number
		K_OPENCL_CHECK(clEnqueueNDRangeKernel(command_queue, accum_noisy_kernel, 2, NULL, k_workset_with_margin_global_size, k_local_size, 0, NULL, NULL));


		#if ENABLE_DEBUG_OUTPUT_TMP_DATA
		if(frame == DEBUG_OUTPUT_FRAME_NUMBER)
		{
			tmpData.features_buffer0.resize(features_buffer_size / sizeof(float));
			K_OPENCL_CHECK(clEnqueueReadBuffer(command_queue, features_buffer, false, 0, features_buffer_size, tmpData.features_buffer0.data(), 0, NULL, NULL));
		}
		#endif

		#if SAVE_INTERMEDIARY_BUFFERS
		SaveDevice3Float32ImageToDisk("noisy_1spp_buffer", frame, command_queue, *noisy_1spp_buffer.current(), noisy1sppBufferDesc);
		#endif
 
		// Phase II: Blockwise Multi-Order Feature Regression (BMFR)
		// -> compute features weightss
		arg_index = 5;
		K_OPENCL_CHECK(clSetKernelArg(fitter_kernel, arg_index++, sizeof(cl_mem), &features_buffer)); // [in] Features buffer (half or single-precision)
		K_OPENCL_CHECK(clSetKernelArg(fitter_kernel, arg_index++, sizeof(cl_int), &frame));  // [in] Current frame number
		K_OPENCL_CHECK(clEnqueueNDRangeKernel(command_queue, fitter_kernel, 1, NULL, k_fitter_global_size, k_fitter_local_size, 0, NULL, NULL));

		// Phase II: Compute noise free color estimate (weighted sum of features)
		arg_index = 3;
		K_OPENCL_CHECK(clSetKernelArg(weighted_sum_kernel, arg_index++, sizeof(cl_mem), normals_buffer.current()));		// [in] Current (world) normals
		K_OPENCL_CHECK(clSetKernelArg(weighted_sum_kernel, arg_index++, sizeof(cl_mem), positions_buffer.current()));	// [in] Current world positions
		K_OPENCL_CHECK(clSetKernelArg(weighted_sum_kernel, arg_index++, sizeof(cl_mem), noisy_1spp_buffer.current()));	// [in] Current noisy 1spp color (only used for debugging)
		K_OPENCL_CHECK(clSetKernelArg(weighted_sum_kernel, arg_index++, sizeof(cl_int), &frame));						// [in] Current frame number
		K_OPENCL_CHECK(clEnqueueNDRangeKernel(command_queue, weighted_sum_kernel, 2, NULL, k_workset_global_size, k_local_size, 0, NULL, NULL));

		#if SAVE_INTERMEDIARY_BUFFERS
		SaveDevice3Float32ImageToDisk("noisefree_1spp", frame, command_queue, noisefree_1spp, noiseFree1sppBufferDesc);
		#endif

		// Phase III: Postprocessing
		// -> accumulate noise-free color estimate + output a tonemapped version
		arg_index = 5;
		K_OPENCL_CHECK(clSetKernelArg(accum_filtered_kernel, arg_index++, sizeof(cl_mem), spp_buffer.current())); // [in] Current number of samples accumulated (for CMA)
		K_OPENCL_CHECK(clSetKernelArg(accum_filtered_kernel, arg_index++, sizeof(cl_mem), noisefree_1spp_accumulated.previous())); // [in]  Previous frame noise-free accumulated color estimate 
		K_OPENCL_CHECK(clSetKernelArg(accum_filtered_kernel, arg_index++, sizeof(cl_mem), noisefree_1spp_accumulated.current()));  // [out] Current frame noise-free accumulated color estimate
		K_OPENCL_CHECK(clSetKernelArg(accum_filtered_kernel, arg_index++, sizeof(cl_int), &frame));	// [in] Current frame number
		K_OPENCL_CHECK(clEnqueueNDRangeKernel(command_queue, accum_filtered_kernel, 2, NULL, k_workset_global_size, k_local_size, 0, NULL, NULL));

		#if SAVE_INTERMEDIARY_BUFFERS
		SaveDevice3Float32ImageToDisk("noisefree_1spp_accumulated", frame, command_queue, *noisefree_1spp_accumulated.current(), noiseFree1sppAccumulatedBufferDesc);
		#endif

		// Phase III: Temporal antialiasing
		arg_index = 2;
		K_OPENCL_CHECK(clSetKernelArg(taa_kernel, arg_index++, sizeof(cl_mem), result_buffer.current()));	// [out] Antialiased frame color buffer
		K_OPENCL_CHECK(clSetKernelArg(taa_kernel, arg_index++, sizeof(cl_mem), result_buffer.previous()));	// [in]  Previous frame color buffer
		K_OPENCL_CHECK(clSetKernelArg(taa_kernel, arg_index++, sizeof(cl_int), &frame));	// [in]  Current frame number
		K_OPENCL_CHECK(clEnqueueNDRangeKernel(command_queue, taa_kernel, 2, NULL, k_workset_global_size, k_local_size, 0, NULL, NULL));

		assert(ret == CL_SUCCESS);

		#if ENABLE_DEBUG_OUTPUT_TMP_DATA
		if(frame == DEBUG_OUTPUT_FRAME_NUMBER)
		{
			tmpData.normals.resize(normals_buffer_size / sizeof(float));
			K_OPENCL_CHECK(clEnqueueReadBuffer(command_queue, *normals_buffer.current(), false, 0, normals_buffer_size, tmpData.normals.data(), 0, NULL, NULL));

			tmpData.positions.resize(positions_buffer_size / sizeof(float));
			K_OPENCL_CHECK(clEnqueueReadBuffer(command_queue, *positions_buffer.current(), false, 0, positions_buffer_size, tmpData.positions.data(), 0, NULL, NULL));

			tmpData.noisy_1spp.resize(noisy_1spp_buffer_size / sizeof(float));
			K_OPENCL_CHECK(clEnqueueReadBuffer(command_queue, *noisy_1spp_buffer.current(), false, 0, noisy_1spp_buffer_size, tmpData.noisy_1spp.data(), 0, NULL, NULL));

			tmpData.prev_frame_pixel_coords_buffer.resize(prev_frame_pixel_coords_buffer_size / sizeof(float));
			K_OPENCL_CHECK(clEnqueueReadBuffer(command_queue, prev_frame_pixel_coords_buffer, false, 0, prev_frame_pixel_coords_buffer_size, tmpData.prev_frame_pixel_coords_buffer.data(), 0, NULL, NULL));

			tmpData.prev_frame_bilinear_samples_validity_mask.resize(prev_frame_bilinear_samples_validity_mask_size / sizeof(unsigned char));
			K_OPENCL_CHECK(clEnqueueReadBuffer(command_queue, prev_frame_bilinear_samples_validity_mask, false, 0, prev_frame_bilinear_samples_validity_mask_size, tmpData.prev_frame_bilinear_samples_validity_mask.data(), 0, NULL, NULL));

			tmpData.spp.resize(spp_buffer_size / sizeof(unsigned char));
			K_OPENCL_CHECK(clEnqueueReadBuffer(command_queue, *spp_buffer.current(), false, 0, spp_buffer_size, tmpData.spp.data(), 0, NULL, NULL));

			tmpData.features_buffer1.resize(features_buffer_size / sizeof(float));
			K_OPENCL_CHECK(clEnqueueReadBuffer(command_queue, features_buffer, false, 0, features_buffer_size, tmpData.features_buffer1.data(), 0, NULL, NULL));

			tmpData.features_weights_buffer.resize(features_weights_buffer_size / sizeof(float));
			K_OPENCL_CHECK(clEnqueueReadBuffer(command_queue, features_weights_buffer, false, 0, features_weights_buffer_size, tmpData.features_weights_buffer.data(), 0, NULL, NULL));

			tmpData.features_min_max_buffer.resize(features_min_max_buffer_size / sizeof(float));
			K_OPENCL_CHECK(clEnqueueReadBuffer(command_queue, features_min_max_buffer, false, 0, features_min_max_buffer_size, tmpData.features_min_max_buffer.data(), 0, NULL, NULL));

			tmpData.noisefree_1spp.resize(noisefree_1spp_size / sizeof(float));
			K_OPENCL_CHECK(clEnqueueReadBuffer(command_queue, noisefree_1spp, false, 0, noisefree_1spp_size, tmpData.noisefree_1spp.data(), 0, NULL, NULL));

			tmpData.noisefree_1spp_accumulated.resize(noisefree_1spp_accumulated_size / sizeof(float));
			K_OPENCL_CHECK(clEnqueueReadBuffer(command_queue, *noisefree_1spp_accumulated.current(), false, 0, noisefree_1spp_accumulated_size, tmpData.noisefree_1spp_accumulated.data(), 0, NULL, NULL));

			tmpData.noisefree_1spp_acc_tonemapped.resize(noisefree_1spp_acc_tonemapped_size / sizeof(float));
			K_OPENCL_CHECK(clEnqueueReadBuffer(command_queue, noisefree_1spp_acc_tonemapped, false, 0, noisefree_1spp_acc_tonemapped_size, tmpData.noisefree_1spp_acc_tonemapped.data(), 0, NULL, NULL));

			tmpData.result.resize(result_buffer_size / sizeof(float));
			K_OPENCL_CHECK(clEnqueueReadBuffer(command_queue, *result_buffer.current(), false, 0, result_buffer_size, tmpData.result.data(), 0, NULL, NULL));

			K_OPENCL_CHECK(clFlush(command_queue));
			K_OPENCL_CHECK(clFinish(command_queue));

			return 0;
		}
		#endif

		SaveDevice3Float32ImageToDisk("result", frame, command_queue, *result_buffer.current(), resultBufferDesc);

		// Swap all double buffers
		std::for_each(all_double_buffers.begin(), all_double_buffers.end(), std::bind(&Double_buffer<cl_mem>::swap, std::placeholders::_1));
	}
    
	// Clean up
	K_OPENCL_CHECK(clFlush(command_queue));
	K_OPENCL_CHECK(clFinish(command_queue));
	K_OPENCL_CHECK(clReleaseKernel(accum_noisy_kernel));
	K_OPENCL_CHECK(clReleaseKernel(fitter_kernel));
	K_OPENCL_CHECK(clReleaseKernel(weighted_sum_kernel));
	K_OPENCL_CHECK(clReleaseKernel(accum_filtered_kernel));
	K_OPENCL_CHECK(clReleaseKernel(taa_kernel));
	K_OPENCL_CHECK(clReleaseProgram(program));
	FreeDoubleBuffer(normals_buffer);
	FreeDoubleBuffer(positions_buffer);
	FreeDoubleBuffer(noisy_1spp_buffer);
	K_OPENCL_CHECK(clReleaseMemObject(features_buffer));
	K_OPENCL_CHECK(clReleaseMemObject(noisefree_1spp));
	FreeDoubleBuffer(noisefree_1spp_accumulated);
	FreeDoubleBuffer(result_buffer);
	K_OPENCL_CHECK(clReleaseMemObject(prev_frame_pixel_coords_buffer));
	K_OPENCL_CHECK(clReleaseMemObject(prev_frame_bilinear_samples_validity_mask));
	K_OPENCL_CHECK(clReleaseMemObject(albedo_buffer));
	K_OPENCL_CHECK(clReleaseMemObject(noisefree_1spp_acc_tonemapped));
	K_OPENCL_CHECK(clReleaseMemObject(features_weights_buffer));
	K_OPENCL_CHECK(clReleaseMemObject(features_min_max_buffer));
	FreeDoubleBuffer(spp_buffer);
	K_OPENCL_CHECK(clReleaseCommandQueue(command_queue));
	K_OPENCL_CHECK(clReleaseContext(context));

	return 0;
}
