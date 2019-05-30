#include "bmfr_cuda.hpp"
#include "bmfr.cuh"

#include "OpenImageIO/imageio.h"
#include <functional>
#include <memory>

#define _CRT_SECURE_NO_WARNINGS
#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

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
#define OUTPUT_FOLDER "outputs/"
#define TO_OUTPUTS_FOLDER(file) OUTPUT_FOLDER ## file
#define OUTPUT_FILE_NAME TO_OUTPUTS_FOLDER("output")

// NOTE: if you want to use other than normal and world_position data you have to make
// it available in the first accumulation kernel and in the weighted sum kernel
#define NOT_SCALED_FEATURE_BUFFERS_STR \
"1.f,"\
"normal.x,"\
"normal.y,"\
"normal.z"\
// The next features are not in the range from -1 to 1 so they are scaled to be from 0 to 1.
#if USE_SCALED_FEATURES
#define SCALED_FEATURE_BUFFERS_STR \
",world_position.x,"\
"world_position.y,"\
"world_position.z,"\
"world_position.x*world_position.x,"\
"world_position.y*world_position.y,"\
"world_position.z*world_position.z"
#else
#define SCALED_FEATURE_BUFFERS_STR ""
#endif

// These local sizes are used with 2D kernels which do not require spesific local size
// (Global sizes are always a multiple of 32)
#define LOCAL_WIDTH 8
#define LOCAL_HEIGHT 8

#define OUTPUT_SIZE (WORKSET_WIDTH * WORKSET_HEIGHT)

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
        template <typename... Args>
        Double_buffer(Args && ... args) : a(std::forward<Args>(args)...), b(std::forward<Args>(args)...), swapped(false){};
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

static inline Operation_result read_image_file(const std::string &file_name, const int frame, float *buffer)
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

static inline Operation_result load_image(float *image, const std::string file_name, const int frame)
{
    Operation_result result = read_image_file(file_name, frame, image);
    if(!result.success)
        return result;

    return {true};
}

static inline float clamp(float value, float minimum, float maximum)
{
    return std::max(std::min(value, maximum), minimum);
}

#define K_CUDA_CHECK(cudaFunc) \
	do { \
		/*printf(#cudaFunc "\n");\*/ \
		cudaError_t ret = cudaFunc; \
		if(ret != cudaSuccess) \
		{ \
			printf("Cuda error: %d\n", ret); \
			__debugbreak(); \
		} \
	} while(0)

struct DeviceBuffer
{
	DeviceBuffer(size_t s)
	{
		K_CUDA_CHECK(cudaMalloc(&data, s));
		size = s;
	}

	~DeviceBuffer()
	{
		K_CUDA_CHECK(cudaFree(data));
	}

	template <typename T>
	T * getTypedData() { return static_cast<T*>(data); }

	template <typename T>
	T const * getTypedData() const { return static_cast<T const*>(data); }

	void * data = nullptr;
	size_t size = 0;
};

int bmfr_cuda(TmpData & tmpData)
{
	std::string features_not_scaled(NOT_SCALED_FEATURE_BUFFERS_STR);
    std::string features_scaled(SCALED_FEATURE_BUFFERS_STR);
    const auto features_not_scaled_count = std::count(features_not_scaled.begin(), features_not_scaled.end(), ',');
    // + 1 because last one does not have ',' after it.
    const auto features_scaled_count = std::count(features_scaled.begin(), features_scaled.end(), ',') + 1;
    // + 3 stands for three noisy spp color channels.
    const auto buffer_count = features_not_scaled_count + features_scaled_count + 3;

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

	// Create CUDA buffers

	printf("\nAllocate CUDA buffers\n");

	const size_t w = IMAGE_WIDTH;
	const size_t h = IMAGE_HEIGHT;

	size_t cudaBufferTotalSize = 0;

	// Albedo buffer (3 * float32) // TODO: compress this
	const BufferDesc albedoBufferDesc = GetAlbedoBufferDesc(w, h);
	const size_t albedo_buffer_size = albedoBufferDesc.byte_size;
    DeviceBuffer albedo_buffer(albedo_buffer_size);
	cudaBufferTotalSize += albedo_buffer_size;

	// (World) normals buffers (3 * float32) in [-1, +1]^3
	const BufferDesc normalsBufferDesc = GetNormalsBufferDesc(w, h);
	const size_t normals_buffer_size = normalsBufferDesc.byte_size;
    Double_buffer<DeviceBuffer> normals_buffer(normals_buffer_size);
	cudaBufferTotalSize += 2 * normals_buffer_size;

	// World positions buffer (3 * float32)
	// TODO: normalize in [0, 1] (or [-1, +1])
	const BufferDesc positionsBufferDesc = GetPositionsBufferDesc(w, h);
	const size_t positions_buffer_size = positionsBufferDesc.byte_size;
    Double_buffer<DeviceBuffer> positions_buffer(positions_buffer_size);
	cudaBufferTotalSize += 2 * positions_buffer_size;
    
	// Noisy 1spp color buffer (3 * float32)
	const BufferDesc noisy1sppBufferDesc = GetNoisy1sppBufferDesc(w, h);
	const size_t noisy_1spp_buffer_size = noisy1sppBufferDesc.byte_size;
	Double_buffer<DeviceBuffer> noisy_1spp_buffer(noisy_1spp_buffer_size);
	cudaBufferTotalSize += 2 * noisy_1spp_buffer_size;

	// Features buffer (half or single-precision) (3 * float16 or 3 * float32)
    const size_t features_buffer_datatype_size = USE_HALF_PRECISION_IN_FEATURES_DATA ? sizeof(short) : sizeof(float);
	const size_t features_buffer_size = WORKSET_WITH_MARGINS_WIDTH * WORKSET_WITH_MARGINS_HEIGHT * buffer_count * features_buffer_datatype_size;
    DeviceBuffer features_buffer(features_buffer_size);
	cudaBufferTotalSize += features_buffer_size;

	// Noise-free color estimate (3 * float32)
	const BufferDesc noiseFree1sppBufferDesc = GetNoiseFree1sppBufferDesc(w, h);
	const size_t noisefree_1spp_size = noiseFree1sppBufferDesc.byte_size;
    DeviceBuffer noisefree_1spp(noisefree_1spp_size);
	cudaBufferTotalSize += noisefree_1spp_size;

	// Noise-free accumulated color estimate (3 * float32)
	const BufferDesc noiseFree1sppAccumulatedBufferDesc = GetNoiseFree1sppAccumulatedBufferDesc(w, h);
	const size_t noisefree_1spp_accumulated_size = noiseFree1sppAccumulatedBufferDesc.byte_size;
    Double_buffer<DeviceBuffer> noisefree_1spp_accumulated(noisefree_1spp_accumulated_size);
	cudaBufferTotalSize += 2 * noisefree_1spp_accumulated_size;

	// Final antialiased color buffer (3 * float32)
	const BufferDesc resultBufferDesc = GetResultBufferDesc(w, h);
	const size_t result_buffer_size = resultBufferDesc.byte_size;
    Double_buffer<DeviceBuffer> result_buffer(result_buffer_size);
	cudaBufferTotalSize += 2 * result_buffer_size;

	// Previous frame pixel coordinates (after reprojection) (2 * float32)
	const BufferDesc prevFramePixelCoordsBufferDesc = GetPrevFramePixelCoordsBufferDesc(w, h);
	const size_t prev_frame_pixel_coords_buffer_size = prevFramePixelCoordsBufferDesc.byte_size;
    DeviceBuffer prev_frame_pixel_coords_buffer(prev_frame_pixel_coords_buffer_size);
	cudaBufferTotalSize += prev_frame_pixel_coords_buffer_size;

	// Validity mask of reprojected bilinear samples into previous frame (uchar 8bits)
	const BufferDesc prevFrameBilinearSamplesValidityMaskBufferDesc = GetPrevFrameBilinearSamplesValidityMaskBufferDesc(w, h);
	const size_t prev_frame_bilinear_samples_validity_mask_size = prevFrameBilinearSamplesValidityMaskBufferDesc.byte_size;
    DeviceBuffer prev_frame_bilinear_samples_validity_mask(prev_frame_bilinear_samples_validity_mask_size);
	cudaBufferTotalSize += prev_frame_bilinear_samples_validity_mask_size;

	// Tonemapped noise-free color estimate w/ albedo (3 * float32)
	const BufferDesc noiseFree1sppAccTonemappedBufferDesc = GetNoiseFree1sppAccTonemappedBufferDesc(w, h);
	const size_t noisefree_1spp_acc_tonemapped_size = noiseFree1sppAccTonemappedBufferDesc.byte_size;
    DeviceBuffer noisefree_1spp_acc_tonemapped(noisefree_1spp_acc_tonemapped_size);
	cudaBufferTotalSize += noisefree_1spp_acc_tonemapped_size;

	// Features weights per color channel (x3) (computed by the BMFR) (3 * float32)
	const size_t features_weights_buffer_size = WORKSET_WITH_MARGIN_BLOCK_COUNT * (buffer_count - 3) * 3 * sizeof(float);
    DeviceBuffer features_weights_buffer(features_weights_buffer_size);
	cudaBufferTotalSize += features_weights_buffer_size;

	// Min and max of features values per block (world_positions) (6 * 2 * float32)
	const size_t features_min_max_buffer_size = WORKSET_WITH_MARGIN_BLOCK_COUNT * 6 * 2 * sizeof(float);
    DeviceBuffer features_min_max_buffer(features_min_max_buffer_size);
	cudaBufferTotalSize += features_min_max_buffer_size;

	// Number of samples accumulated (for cumulative moving average) (char 8bits)
	const BufferDesc sppBufferDesc = GetSppBufferDesc(w, h);
	const size_t spp_buffer_size = sppBufferDesc.byte_size;
    Double_buffer<DeviceBuffer> spp_buffer(spp_buffer_size);
	cudaBufferTotalSize += 2 * spp_buffer_size;

	std::vector<Double_buffer<DeviceBuffer> *> all_double_buffers =
	{
		&normals_buffer,
		&positions_buffer,
		&noisy_1spp_buffer,
		&noisefree_1spp_accumulated,
		&result_buffer,
		&spp_buffer
	};

	printf("CUDA buffers total size: %.3fMB\n", float(cudaBufferTotalSize) / 1024.f / 1024.f);

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
    const auto r_size = ((buffer_count - 2) * (buffer_count - 1) / 2) * sizeof(vec3);
	#else
    const auto r_size = (buffer_count - 2) * (buffer_count - 2) * sizeof(vec3);
	#endif

    printf("\nRun kernels.\n");


	// Only work with 3 channels float32 image
	const auto SaveDevice3Float32ImageToDisk = [](
		std::string const & filename,
		int frame,
		DeviceBuffer const & buffer,
		BufferDesc const & desc
	)
	{
		assert(buffer.size == desc.w * desc.h * 3 * sizeof(float));
		const size_t datasize = desc.byte_size;
		const size_t numelem  = datasize / sizeof(float);
		std::vector<float> outdata;
		outdata.resize(numelem);

		K_CUDA_CHECK(cudaMemcpy(outdata.data(), buffer.data, datasize, cudaMemcpyDeviceToHost));
		K_CUDA_CHECK(cudaDeviceSynchronize());

		std::string output_filename = OUTPUT_FOLDER + filename + "_" + std::to_string(frame) + "_cuda.png";

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
	};

    dim3 k_block_size(LOCAL_WIDTH, LOCAL_HEIGHT);
    dim3 k_workset_grid_size((WORKSET_WIDTH + k_block_size.x - 1) / k_block_size.x, (WORKSET_HEIGHT + k_block_size.y - 1) / k_block_size.y);
	dim3 k_workset_with_margin_grid_size((WORKSET_WITH_MARGINS_WIDTH + k_block_size.x - 1) / k_block_size.x, (WORKSET_WITH_MARGINS_HEIGHT + k_block_size.y - 1) / k_block_size.y);
    dim3 k_fitter_block_size(LOCAL_SIZE);
    dim3 k_fitter_grid_size((FITTER_KERNEL_GLOBAL_RANGE + k_fitter_block_size.x - 1) / k_fitter_block_size.x);
	
	for(int frame = 0; frame < FRAME_COUNT; ++frame)
    {
		printf("Frame %d\n", frame);

		printf("  Load frame input data from disk\n");
		LoadFrameInputData(frameInput, frame);

		printf("  Transfert data from host to device\n");
		K_CUDA_CHECK(cudaMemcpy(albedo_buffer.data, frameInput.albedos.data(), frameInput.albedos.size() * sizeof(float), cudaMemcpyHostToDevice));
        K_CUDA_CHECK(cudaMemcpy(normals_buffer.current()->data, frameInput.normals.data(), frameInput.normals.size() * sizeof(float), cudaMemcpyHostToDevice));
        K_CUDA_CHECK(cudaMemcpy(positions_buffer.current()->data, frameInput.positions.data(), frameInput.positions.size() * sizeof(float), cudaMemcpyHostToDevice));
        K_CUDA_CHECK(cudaMemcpy(noisy_1spp_buffer.current()->data, frameInput.noisy1spps.data(), frameInput.noisy1spps.size() * sizeof(float), cudaMemcpyHostToDevice));

		// Phase I:
		//  - accumulate noisy 1spp
		//  - compute previous frame pixel coordinates (after reprojection)
		//  - generate validity bit mask of bilinear samples of previous frame
		//  - concatenate the different features in a single buffer
        // Note: On the first frame accum_noisy_kernel just copies to the features_buffer
        
		printf("  Run accumulate_noisy_data kernel: grid (%d, %d) | block(%d, %d)\n", 
			k_workset_with_margin_grid_size.x,
			k_workset_with_margin_grid_size.y,
			k_block_size.x,
			k_block_size.y
		);

        const int matrix_index = frame == 0 ? 0 : frame - 1;
		const mat4x4 cam_mat = *reinterpret_cast<mat4x4 const *>(&camera_matrices[matrix_index][0][0]);
		const vec2 pix_off = *reinterpret_cast<vec2 const *>(&pixel_offsets[frame][0]);
		run_accumulate_noisy_data(
			k_workset_with_margin_grid_size,
			k_block_size,
			prev_frame_pixel_coords_buffer.getTypedData<vec2>(),
			prev_frame_bilinear_samples_validity_mask.getTypedData<unsigned char>(),
			normals_buffer.current()->getTypedData<float>(),
			normals_buffer.previous()->getTypedData<float>(),
			positions_buffer.current()->getTypedData<float>(),
			positions_buffer.previous()->getTypedData<float>(),
			noisy_1spp_buffer.current()->getTypedData<float>(),
			noisy_1spp_buffer.previous()->getTypedData<float>(),
			// TODO: invert the order of the spp buffers
			spp_buffer.previous()->getTypedData<unsigned char>(),
			spp_buffer.current()->getTypedData<unsigned char>(),
			features_buffer.getTypedData<float>(),
			cam_mat,
			pix_off,
			frame
		);

		K_CUDA_CHECK(cudaDeviceSynchronize());

		#if SAVE_INTERMEDIARY_BUFFERS
		SaveDevice3Float32ImageToDisk("noisy_1spp_buffer", frame, *noisy_1spp_buffer.current(), noisy1sppBufferDesc);
		#endif

		// Phase II: Blockwise Multi-Order Feature Regression (BMFR)
		// -> compute features weights

		if(0)
		{
			size_t features_buffer_size = features_buffer.size;
			std::vector<float> debugData;
			debugData.resize(features_buffer_size / sizeof(float));

			for(auto i = 0; i < debugData.size(); ++i)
			{
				debugData[i] = 0.42f;
			}
			K_CUDA_CHECK(cudaMemcpy(features_buffer.data, debugData.data(), features_buffer_size, cudaMemcpyHostToDevice));
			K_CUDA_CHECK(cudaDeviceSynchronize());
		}

		printf("  Run fitter kernel: grid (%d, %d) | block(%d, %d)\n", 
			k_fitter_grid_size.x,
			k_fitter_grid_size.y,
			k_fitter_block_size.x,
			k_fitter_block_size.y
		);

		run_fitter(
			k_fitter_grid_size,
			k_fitter_block_size,
			features_weights_buffer.getTypedData<float>(),
			features_min_max_buffer.getTypedData<float>(),
			features_buffer.getTypedData<float>(),
			frame
		);

		K_CUDA_CHECK(cudaDeviceSynchronize());

		// Phase II: Compute noise free color estimate (weighted sum of features)

		printf("  Run weighted_sum kernel: grid (%d, %d) | block(%d, %d)\n", 
			k_workset_grid_size.x,
			k_workset_grid_size.y,
			k_block_size.x,
			k_block_size.y
		);

		run_weighted_sum(
			k_workset_grid_size,
			k_block_size,
			features_weights_buffer.getTypedData<float>(),
			features_min_max_buffer.getTypedData<float>(),
			noisefree_1spp.getTypedData<float>(),
			normals_buffer.current()->getTypedData<float>(),
			positions_buffer.current()->getTypedData<float>(),
			noisy_1spp_buffer.current()->getTypedData<float>(),
			frame
		);

		K_CUDA_CHECK(cudaDeviceSynchronize());

		#if SAVE_INTERMEDIARY_BUFFERS
		SaveDevice3Float32ImageToDisk("noisefree_1spp", frame, noisefree_1spp, noiseFree1sppBufferDesc);
		#endif

		// Phase III: Postprocessing
		// -> accumulate noise-free color estimate + output a tonemapped version

		printf("  Run accumulate_filtered_data kernel: grid (%d, %d) | block(%d, %d)\n", 
			k_workset_grid_size.x,
			k_workset_grid_size.y,
			k_block_size.x,
			k_block_size.y
		);

		run_accumulate_filtered_data(
			k_workset_grid_size,
			k_block_size,
			noisefree_1spp.getTypedData<float>(),
			prev_frame_pixel_coords_buffer.getTypedData<vec2>(),
			prev_frame_bilinear_samples_validity_mask.getTypedData<unsigned char>(),
			albedo_buffer.getTypedData<float>(),
			noisefree_1spp_acc_tonemapped.getTypedData<float>(),
			spp_buffer.current()->getTypedData<unsigned char>(),
			noisefree_1spp_accumulated.previous()->getTypedData<float>(), 
			noisefree_1spp_accumulated.current()->getTypedData<float>(),
			frame
		);

		#if SAVE_INTERMEDIARY_BUFFERS
		SaveDevice3Float32ImageToDisk("noisefree_1spp_accumulated", frame, *noisefree_1spp_accumulated.current(), noiseFree1sppAccumulatedBufferDesc);
		#endif

		// Phase III: Temporal antialiasing

		printf("  Run taa kernel: grid (%d, %d) | block(%d, %d)\n", 
			k_workset_grid_size.x,
			k_workset_grid_size.y,
			k_block_size.x,
			k_block_size.y
		);

		run_taa(
			k_workset_grid_size,
			k_block_size,
			prev_frame_pixel_coords_buffer.getTypedData<vec2>(),
			noisefree_1spp_acc_tonemapped.getTypedData<float>(),
			result_buffer.current()->getTypedData<float>(),
			result_buffer.previous()->getTypedData<float>(),
			frame
		);

		#if ENABLE_DEBUG_OUTPUT_TMP_DATA
		if(frame == DEBUG_OUTPUT_FRAME_NUMBER)
		{
			tmpData.normals.resize(normals_buffer_size / sizeof(float));
			K_CUDA_CHECK(cudaMemcpy(tmpData.normals.data(), normals_buffer.current()->data, normals_buffer_size, cudaMemcpyDeviceToHost));

			tmpData.positions.resize(positions_buffer_size / sizeof(float));
			K_CUDA_CHECK(cudaMemcpy(tmpData.positions.data(), positions_buffer.current()->data, positions_buffer_size, cudaMemcpyDeviceToHost));

			tmpData.noisy_1spp.resize(noisy_1spp_buffer_size / sizeof(float));
			K_CUDA_CHECK(cudaMemcpy(tmpData.noisy_1spp.data(), noisy_1spp_buffer.current()->data, noisy_1spp_buffer_size, cudaMemcpyDeviceToHost));

			tmpData.prev_frame_pixel_coords_buffer.resize(prev_frame_pixel_coords_buffer_size / sizeof(float));
			K_CUDA_CHECK(cudaMemcpy(tmpData.prev_frame_pixel_coords_buffer.data(), prev_frame_pixel_coords_buffer.data, prev_frame_pixel_coords_buffer_size, cudaMemcpyDeviceToHost));

			tmpData.prev_frame_bilinear_samples_validity_mask.resize(prev_frame_bilinear_samples_validity_mask_size / sizeof(unsigned char));
			K_CUDA_CHECK(cudaMemcpy(tmpData.prev_frame_bilinear_samples_validity_mask.data(), prev_frame_bilinear_samples_validity_mask.data, prev_frame_bilinear_samples_validity_mask_size, cudaMemcpyDeviceToHost));

			tmpData.spp.resize(spp_buffer_size / sizeof(unsigned char));
			K_CUDA_CHECK(cudaMemcpy(tmpData.spp.data(), spp_buffer.current()->data, spp_buffer_size, cudaMemcpyDeviceToHost));

			tmpData.features_buffer.resize(features_buffer_size / sizeof(float));
			K_CUDA_CHECK(cudaMemcpy(tmpData.features_buffer.data(), features_buffer.data, features_buffer_size, cudaMemcpyDeviceToHost));

			tmpData.features_weights_buffer.resize(features_weights_buffer_size / sizeof(float));
			K_CUDA_CHECK(cudaMemcpy(tmpData.features_weights_buffer.data(), features_weights_buffer.data, features_weights_buffer_size, cudaMemcpyDeviceToHost));

			tmpData.features_min_max_buffer.resize(features_min_max_buffer_size / sizeof(float));
			K_CUDA_CHECK(cudaMemcpy(tmpData.features_min_max_buffer.data(), features_min_max_buffer.data, features_min_max_buffer_size, cudaMemcpyDeviceToHost));

			tmpData.noisefree_1spp.resize(noisefree_1spp_size / sizeof(float));
			K_CUDA_CHECK(cudaMemcpy(tmpData.noisefree_1spp.data(), noisefree_1spp.data, noisefree_1spp_size, cudaMemcpyDeviceToHost));

			tmpData.noisefree_1spp_accumulated.resize(noisefree_1spp_accumulated_size / sizeof(float));
			K_CUDA_CHECK(cudaMemcpy(tmpData.noisefree_1spp_accumulated.data(), noisefree_1spp_accumulated.current()->data, noisefree_1spp_accumulated_size, cudaMemcpyDeviceToHost));

			tmpData.noisefree_1spp_acc_tonemapped.resize(noisefree_1spp_acc_tonemapped_size / sizeof(float));
			K_CUDA_CHECK(cudaMemcpy(tmpData.noisefree_1spp_acc_tonemapped.data(), noisefree_1spp_acc_tonemapped.data, noisefree_1spp_acc_tonemapped_size, cudaMemcpyDeviceToHost));

			tmpData.result.resize(result_buffer_size / sizeof(float));
			K_CUDA_CHECK(cudaMemcpy(tmpData.result.data(), result_buffer.current()->data, result_buffer_size, cudaMemcpyDeviceToHost));

			K_CUDA_CHECK(cudaDeviceSynchronize());
			
			return 0;
		}
		#endif

		SaveDevice3Float32ImageToDisk("result", frame, *result_buffer.current(), resultBufferDesc);

		// Swap all double buffers
        std::for_each(all_double_buffers.begin(), all_double_buffers.end(), std::bind(&Double_buffer<DeviceBuffer>::swap, std::placeholders::_1));
	}
	return 0;
}
