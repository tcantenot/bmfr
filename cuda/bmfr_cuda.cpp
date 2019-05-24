
#include "bmfr_cuda.hpp"
#include "bmfr.cuh"

#include "OpenImageIO/imageio.h"
#include <functional>
#include <memory>
#include <fstream>

#define _CRT_SECURE_NO_WARNINGS
#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

// TODO detect FRAME_COUNT from the input files
#define FRAME_COUNT 1
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
#define TO_OUTPUTS_FOLDER(file) "outputs/" ## file
#define OUTPUT_FILE_NAME TO_OUTPUTS_FOLDER("output")

// NOTE: if you want to use other than normal and world_position data you have to make
// it available in the first accumulation kernel and in the weighted sum kernel
#define NOT_SCALED_FEATURE_BUFFERS_STR \
"1.f,"\
"normal.x,"\
"normal.y,"\
"normal.z,"\
// The next features are not in the range from -1 to 1 so they are scaled to be from 0 to 1.
#if 0
#define SCALED_FEATURE_BUFFERS_STR \
"world_position.x,"\
"world_position.y,"\
"world_position.z,"\
"world_position.x*world_position.x,"\
"world_position.y*world_position.y,"\
"world_position.z*world_position.z"
#else
#define SCALED_FEATURE_BUFFERS_STR ""
#endif
// TODO: not sure that the buffers must be OUTPUT_SIZE: IMAGE_WITDH * IMAGE_HEIGHT should be enough
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

static void bmfr()
{
	std::string features_not_scaled(NOT_SCALED_FEATURE_BUFFERS_STR);
    std::string features_scaled(SCALED_FEATURE_BUFFERS_STR);
    const auto features_not_scaled_count = std::count(features_not_scaled.begin(), features_not_scaled.end(), ',');
    // + 1 because last one does not have ',' after it.
    const auto features_scaled_count = std::count(features_scaled.begin(), features_scaled.end(), ',') + 1;
    // + 3 stands for three noisy spp color channels.
    const auto buffer_count = features_not_scaled_count + features_scaled_count + 3;

	// Load input data arrays from disk to host memory
	printf("Loading input data.\n");
	std::vector<float> out_data[FRAME_COUNT];
	std::vector<float> albedos[FRAME_COUNT];
	std::vector<float> normals[FRAME_COUNT];
	std::vector<float> positions[FRAME_COUNT];
	std::vector<float> noisy_input[FRAME_COUNT];
	bool error = false;
	#pragma omp parallel for
    for(int frame = 0; frame < FRAME_COUNT; ++frame)
    {
        if(error)
            continue;

		printf("  Loading data of frame %d\n", frame);

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
        return;
    }

	printf("Done loading data.\n");

	// Create CUDA buffers

	printf("\nAllocate CUDA buffers\n");

	size_t cudaBufferTotalSize = 0;

	// (World) normals buffers (3 * float32) in [-1, +1]^3
	// TODO: compress data? half? -> would require different storage than feature buffer
    Double_buffer<DeviceBuffer> normals_buffer(OUTPUT_SIZE * 3 * sizeof(float));
	cudaBufferTotalSize += 2 * normals_buffer.current()->size;

	// World positions buffer (3 * float32)
	// TODO: normalize in [0, 1] (or [-1, +1])
    Double_buffer<DeviceBuffer> positions_buffer(OUTPUT_SIZE * 3 * sizeof(float));
	cudaBufferTotalSize += 2 * positions_buffer.current()->size;
    
	// Noisy 1spp color buffer (3 * float32)
	Double_buffer<DeviceBuffer> noisy_1spp_color_buffer(OUTPUT_SIZE * 3 * sizeof(float));
	cudaBufferTotalSize += 2 * noisy_1spp_color_buffer.current()->size;

	// Features buffer size
    size_t features_buffer_datatype_size = USE_HALF_PRECISION_IN_FEATURES_DATA ? sizeof(short) : sizeof(float);

	// Features buffer (half or single-precision) (3 * float16 or 3 * float32)
    DeviceBuffer features_buffer(WORKSET_WITH_MARGINS_WIDTH * WORKSET_WITH_MARGINS_HEIGHT * buffer_count * features_buffer_datatype_size);
	cudaBufferTotalSize += features_buffer.size;

	// Noise-free color estimate (3 * float32)
    DeviceBuffer noisefree_color_estimate(OUTPUT_SIZE * 3 * sizeof(float));
	cudaBufferTotalSize += noisefree_color_estimate.size;

	// TODO: why does the size of this buffer is WORKSET_WITH_MARGINS_WIDTH x WORKSET_WITH_MARGINS_HEIGHT and not WORKSET_WIDTH x WORKSET_HEIGHT?
	// -> it seems we are not writing/reading in the part of the buffer that is outside of image (because of offsets)
	// (see kernel 'accumulate_filtered_data' that is the only kernel using it)
	// --> should have the same size as 'tone_mapped_buffer'
	// Noise-free accumulated color estimate (3 * float32)
    Double_buffer<DeviceBuffer> noisefree_accumulated_color_estimate(WORKSET_WITH_MARGINS_WIDTH * WORKSET_WITH_MARGINS_HEIGHT * 3 * sizeof(float));
	cudaBufferTotalSize += 2 * noisefree_accumulated_color_estimate.current()->size;

	// Final antialiased color buffer (3 * float32)
    Double_buffer<DeviceBuffer> result_buffer(OUTPUT_SIZE * 3 * sizeof(float));
	cudaBufferTotalSize += 2 * result_buffer.current()->size;

	// Previous frame pixel coordinates (after reprojection) (2 * float32)
    DeviceBuffer prev_frame_pixel_coords_buffer(OUTPUT_SIZE * sizeof(vec2));
	cudaBufferTotalSize += prev_frame_pixel_coords_buffer.size;

	// Validity mask of reprojected bilinear samples into previous frame (uchar 8bits)
    DeviceBuffer prev_frame_bilinear_samples_validity_mask(OUTPUT_SIZE * sizeof(unsigned char));
	cudaBufferTotalSize += prev_frame_bilinear_samples_validity_mask.size;

	// Albedo buffer (3 * float32) // TODO: compress this
    DeviceBuffer albedo_buffer(IMAGE_WIDTH * IMAGE_HEIGHT * 3 * sizeof(float));
	cudaBufferTotalSize += albedo_buffer.size;

	// Tonemapped noise-free color estimate w/ albedo (3 * float32)
    DeviceBuffer tone_mapped_buffer(IMAGE_WIDTH * IMAGE_HEIGHT * 3 * sizeof(float));
	cudaBufferTotalSize += tone_mapped_buffer.size;

	// Features weights per color channel (x3) (computed by the BMFR) (3 * float32)
    //DeviceBuffer features_weights_buffer((FITTER_KERNEL_GLOBAL_RANGE / 256) * (buffer_count - 3) * 3 * sizeof(float));
    DeviceBuffer features_weights_buffer(WORKSET_WITH_MARGIN_BLOCK_COUNT * (buffer_count - 3) * 3 * sizeof(float));
	cudaBufferTotalSize += features_weights_buffer.size;

	// Min and max of features values per block (world_positions) (6 * 2 * float32)
    //DeviceBuffer features_min_max_buffer((FITTER_KERNEL_GLOBAL_RANGE / 256) * 6 * sizeof(vec2));
    DeviceBuffer features_min_max_buffer(WORKSET_WITH_MARGIN_BLOCK_COUNT * 6 * sizeof(vec2));
	cudaBufferTotalSize += features_min_max_buffer.size;

	// Number of samples accumulated (for cumulative moving average) (char 8bits)
    Double_buffer<DeviceBuffer> spp_buffer(OUTPUT_SIZE * sizeof(char));
	cudaBufferTotalSize += 2 * spp_buffer.current()->size;

	std::vector<Double_buffer<DeviceBuffer> *> all_double_buffers =
	{
		&normals_buffer,
		&positions_buffer,
		&noisy_1spp_color_buffer,
		&noisefree_accumulated_color_estimate,
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

    printf("\nRun and profile kernels.\n");

	// Only work with 3 channels float32 image
	const auto SaveDevice3Float32ImageToDisk = [](DeviceBuffer const & buffer, std::string const & output_file_name)
	{
		const size_t datasize = OUTPUT_SIZE * 3 * sizeof(float);
		std::vector<float> outdata;
		outdata.resize(datasize);

		K_CUDA_CHECK(cudaMemcpy(outdata.data(), buffer.data, datasize, cudaMemcpyDeviceToHost));
		K_CUDA_CHECK(cudaDeviceSynchronize());

		 // Output image
		printf("  Save image %s\n", output_file_name.c_str());

        // Crops back from WORKSET_SIZE to IMAGE_SIZE
        OpenImageIO::ImageSpec spec(IMAGE_WIDTH, IMAGE_HEIGHT, 3, OpenImageIO::TypeDesc::FLOAT);
        std::unique_ptr<OpenImageIO::ImageOutput> out(OpenImageIO::ImageOutput::create(output_file_name));
        if(out && out->open(output_file_name, spec))
        {
            out->write_image(OpenImageIO::TypeDesc::FLOAT, outdata.data(), 3 * sizeof(float), WORKSET_WIDTH * 3 * sizeof(float), 0);
            out->close();
        }
        else
        {
            printf("  Can't create image file on disk to location %s\n", output_file_name.c_str());
        }
	};

	const auto SaveDeviceFeaturesMinMaxImageToDisk = [](DeviceBuffer const & buffer, std::string const & output_file_name)
	{
		const size_t datasize = OUTPUT_SIZE * 3 * sizeof(float);
		std::vector<float> outdata;
		outdata.resize(datasize);

		K_CUDA_CHECK(cudaMemcpy(outdata.data(), buffer.data, datasize, cudaMemcpyDeviceToHost));
		K_CUDA_CHECK(cudaDeviceSynchronize());

			// Output image
		printf("  Save image %s\n", output_file_name.c_str());

		// Crops back from WORKSET_SIZE to IMAGE_SIZE
		OpenImageIO::ImageSpec spec(IMAGE_WIDTH, IMAGE_HEIGHT, 3, OpenImageIO::TypeDesc::FLOAT);
		std::unique_ptr<OpenImageIO::ImageOutput> out(OpenImageIO::ImageOutput::create(output_file_name));
		if(out && out->open(output_file_name, spec))
		{
			out->write_image(OpenImageIO::TypeDesc::FLOAT, outdata.data(), 3 * sizeof(float), WORKSET_WIDTH * 3 * sizeof(float), 0);
			out->close();
		}
		else
		{
			printf("  Can't create image file on disk to location %s\n", output_file_name.c_str());
		}
	};

    dim3 k_workset_grid_size((IMAGE_WIDTH + BLOCK_EDGE_LENGTH - 1) / BLOCK_EDGE_LENGTH, (IMAGE_HEIGHT + BLOCK_EDGE_LENGTH - 1) / BLOCK_EDGE_LENGTH);
	dim3 k_workset_with_margin_grid_size(k_workset_grid_size.x + 1, k_workset_grid_size.y + 1);
    dim3 k_block_size(LOCAL_WIDTH, LOCAL_HEIGHT);
    dim3 k_fitter_grid_size(WORKSET_WITH_MARGIN_BLOCK_COUNT);
    dim3 k_fitter_block_size(LOCAL_SIZE);

	for(int frame = 0; frame < FRAME_COUNT; ++frame)
    {
		printf("Frame %d\n", frame);

		printf("  Transfert data from host to device\n");
		K_CUDA_CHECK(cudaMemcpy(albedo_buffer.data, albedos[frame].data(), IMAGE_WIDTH * IMAGE_HEIGHT * 3 * sizeof(float), cudaMemcpyHostToDevice));
        K_CUDA_CHECK(cudaMemcpy(normals_buffer.current()->data, normals[frame].data(), IMAGE_WIDTH * IMAGE_HEIGHT * 3 * sizeof(float), cudaMemcpyHostToDevice));
        K_CUDA_CHECK(cudaMemcpy(positions_buffer.current()->data, positions[frame].data(), IMAGE_WIDTH * IMAGE_HEIGHT * 3 * sizeof(float), cudaMemcpyHostToDevice));
        K_CUDA_CHECK(cudaMemcpy(noisy_1spp_color_buffer.current()->data, noisy_input[frame].data(), IMAGE_WIDTH * IMAGE_HEIGHT * 3 * sizeof(float), cudaMemcpyHostToDevice));

		printf("  Run accumulate_noisy_data kernel: grid (%d, %d) | block(%d, %d)\n", 
			k_workset_with_margin_grid_size.x,
			k_workset_with_margin_grid_size.y,
			k_block_size.x,
			k_block_size.y
		);

        const int matrix_index = frame == 0 ? 0 : frame - 1;
		run_accumulate_noisy_data(
			k_workset_with_margin_grid_size,
			k_block_size,
			prev_frame_pixel_coords_buffer.getTypedData<vec2>(),
			prev_frame_bilinear_samples_validity_mask.getTypedData<unsigned char>(),
			normals_buffer.current()->getTypedData<float>(),
			normals_buffer.previous()->getTypedData<float>(),
			positions_buffer.current()->getTypedData<float>(),
			positions_buffer.previous()->getTypedData<float>(),
			noisy_1spp_color_buffer.current()->getTypedData<float>(),
			noisy_1spp_color_buffer.previous()->getTypedData<float>(),
			spp_buffer.current()->getTypedData<unsigned char>(),
			spp_buffer.previous()->getTypedData<unsigned char>(),
			features_buffer.getTypedData<float>(),
			*reinterpret_cast<mat4x4 const *>(&camera_matrices[matrix_index][0][0]),
			*reinterpret_cast<vec2 const *>(&pixel_offsets[frame][0]),
			frame
		);

		K_CUDA_CHECK(cudaDeviceSynchronize());

		//SaveDevice3Float32ImageToDisk(*normals_buffer.current(),	TO_OUTPUTS_FOLDER("normals_buffer_") + std::to_string(frame) + ".png");
		//SaveDevice3Float32ImageToDisk(*positions_buffer.previous(), TO_OUTPUTS_FOLDER("positions_buffer_") + std::to_string(frame) + ".png");
		//SaveDevice3Float32ImageToDisk(*noisy_1spp_color_buffer.current(),  TO_OUTPUTS_FOLDER("noisy_1spp_color_buffer_") + std::to_string(frame) + ".png");

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
			noisefree_color_estimate.getTypedData<float>(),
			normals_buffer.current()->getTypedData<float>(),
			positions_buffer.current()->getTypedData<float>(),
			noisy_1spp_color_buffer.current()->getTypedData<float>(),
			frame
		);

		K_CUDA_CHECK(cudaDeviceSynchronize());

		//SaveDevice3Float32ImageToDisk(features_min_max_buffer, TO_OUTPUTS_FOLDER("features_min_max_buffer_") + std::to_string(frame) + ".png");
		SaveDevice3Float32ImageToDisk(noisefree_color_estimate, TO_OUTPUTS_FOLDER("noisefree_color_estimate_") + std::to_string(frame) + ".png");

		// This is not timed because in real use case the result is stored to frame buffer
        //K_CUDA_CHECK(cudaMemcpy(out_data[frame].data(), result_buffer.current()->data, OUTPUT_SIZE * 3 * sizeof(float), cudaMemcpyDeviceToHost));
		//K_CUDA_CHECK(cudaDeviceSynchronize());

		// Swap all double buffers
        std::for_each(all_double_buffers.begin(), all_double_buffers.end(), std::bind(&Double_buffer<DeviceBuffer>::swap, std::placeholders::_1));
	}

	return;

	// Store results
    error = false;
	#pragma omp parallel for
    for(int frame = 0; frame < FRAME_COUNT; ++frame)
    {
        if(error)
            continue;

        // Output image
        std::string output_file_name = OUTPUT_FILE_NAME + std::to_string(frame) + "_cuda.png";

		printf("Save image %s\n", output_file_name.c_str());

        // Crops back from WORKSET_SIZE to IMAGE_SIZE
        OpenImageIO::ImageSpec spec(IMAGE_WIDTH, IMAGE_HEIGHT, 3, OpenImageIO::TypeDesc::FLOAT);
        std::unique_ptr<OpenImageIO::ImageOutput> out(OpenImageIO::ImageOutput::create(output_file_name));
        if(out && out->open(output_file_name, spec))
        {
            out->write_image(OpenImageIO::TypeDesc::FLOAT, out_data[frame].data(), 3 * sizeof(float), WORKSET_WIDTH * 3 * sizeof(float), 0);
            out->close();
        }
        else
        {
            printf("Can't create image file on disk to location %s\n", output_file_name.c_str());
            error = true;
            continue;
        }
    }

    if(error)
    {
        printf("One or more errors occurred during image saving\n");
        return;
    }
}

int bmfr_cuda()
{
	bmfr();
	//run_cuda_hello();
	return 0;
}
