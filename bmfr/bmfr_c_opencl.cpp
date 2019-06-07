#include "bmfr_c_opencl.hpp"
#include "utils.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <functional>
#include <CL/cl.h>


void init_opencl(cl_context & context, cl_command_queue & command_queue, cl_device_id & device)
{
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
			LOG("\n %d. Platform \n", i+1);

			for (j = 0; j < attributeCount; j++) {

				// get platform attribute value size
				clGetPlatformInfo(platforms[i], attributeTypes[j], 0, NULL, &infoSize);
				info = (char*) malloc(infoSize);

				// get platform attribute value
				clGetPlatformInfo(platforms[i], attributeTypes[j], infoSize, info, NULL);

				LOG("  %d.%d %-11s: %s\n", i+1, j+1, attributeNames[j], info);
				free(info);
			}

			LOG("\n");

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
				LOG("%d. Device: %s\n", j+1, value);
				free(value);

				// print hardware device version
				clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, 0, NULL, &valueSize);
				value = (char*) malloc(valueSize);
				clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, valueSize, value, NULL);
				LOG(" %d.%d Hardware version: %s\n", j+1, 1, value);
				free(value);

				// print software driver version
				clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, 0, NULL, &valueSize);
				value = (char*) malloc(valueSize);
				clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, valueSize, value, NULL);
				LOG(" %d.%d Software version: %s\n", j+1, 2, value);
				free(value);

				// print c version supported by compiler for device
				clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &valueSize);
				value = (char*) malloc(valueSize);
				clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, valueSize, value, NULL);
				LOG(" %d.%d OpenCL C version: %s\n", j+1, 3, value);
				free(value);

				// print parallel compute units
				clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS,
						sizeof(maxComputeUnits), &maxComputeUnits, NULL);
				LOG(" %d.%d Parallel compute units: %d\n", j+1, 4, maxComputeUnits);

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
    LOG("\nSelected device: %s\n", value);
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
    LOG("Selected device: %s\n", value);
    free(value);
#endif

	cl_int ret = CL_SUCCESS;

	// Create an OpenCL context
	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
 
	// Create a command queue
	command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

	device = device_id;
}

void init_bmfr_opencl_buffers(BMFROpenCLBuffers & buffers, size_t w, size_t h, size_t features_count, cl_context c)
{
	size_t openCLBuffersTotalSize = 0;

	// Albedo buffer (3 * float32) // TODO: compress this
	const BufferDesc albedoBufferDesc = GetAlbedoBufferDesc(w, h);
	const size_t albedo_buffer_size = albedoBufferDesc.byte_size;
    buffers.albedo_buffer.init(c, CL_MEM_READ_ONLY, albedo_buffer_size);
	openCLBuffersTotalSize += albedo_buffer_size;

	// (World) normals buffers (3 * float32) in [-1, +1]^3
	const BufferDesc normalsBufferDesc = GetNormalsBufferDesc(w, h);
	const size_t normals_buffer_size = normalsBufferDesc.byte_size;
	buffers.normals_buffer[0].init(c, CL_MEM_READ_WRITE, normals_buffer_size);
	buffers.normals_buffer[1].init(c, CL_MEM_READ_WRITE, normals_buffer_size);
	openCLBuffersTotalSize += 2 * normals_buffer_size;

	// World positions buffer (3 * float32)
	// TODO: normalize in [0, 1] (or [-1, +1])
	const BufferDesc positionsBufferDesc = GetPositionsBufferDesc(w, h);
	const size_t positions_buffer_size = positionsBufferDesc.byte_size;
    buffers.positions_buffer[0].init(c, CL_MEM_READ_WRITE, positions_buffer_size);
    buffers.positions_buffer[1].init(c, CL_MEM_READ_WRITE, positions_buffer_size);
	openCLBuffersTotalSize += 2 * positions_buffer_size;
    
	// Noisy 1spp color buffer (3 * float32)
	const BufferDesc noisy1sppBufferDesc = GetNoisy1sppBufferDesc(w, h);
	const size_t noisy_1spp_buffer_size = noisy1sppBufferDesc.byte_size;
	buffers.noisy_1spp_buffer[0].init(c, CL_MEM_READ_WRITE, noisy_1spp_buffer_size);
	buffers.noisy_1spp_buffer[1].init(c, CL_MEM_READ_WRITE, noisy_1spp_buffer_size);
	openCLBuffersTotalSize += 2 * noisy_1spp_buffer_size;

	// Features buffer (half or single-precision) (3 * float16 or 3 * float32)
    const size_t features_buffer_datatype_size = USE_HALF_PRECISION_IN_FEATURES_DATA ? sizeof(short) : sizeof(float);
	const size_t features_buffer_size = WORKSET_WITH_MARGINS_WIDTH * WORKSET_WITH_MARGINS_HEIGHT * features_count * features_buffer_datatype_size;
    buffers.features_buffer.init(c, CL_MEM_READ_WRITE, features_buffer_size);
	openCLBuffersTotalSize += features_buffer_size;

	// Noise-free color estimate (3 * float32)
	const BufferDesc noiseFree1sppBufferDesc = GetNoiseFree1sppBufferDesc(w, h);
	const size_t noisefree_1spp_size = noiseFree1sppBufferDesc.byte_size;
    buffers.noisefree_1spp.init(c, CL_MEM_READ_WRITE, noisefree_1spp_size);
	openCLBuffersTotalSize += noisefree_1spp_size;

	// Noise-free accumulated color estimate (3 * float32)
	const BufferDesc noiseFree1sppAccumulatedBufferDesc = GetNoiseFree1sppAccumulatedBufferDesc(w, h);
	const size_t noisefree_1spp_accumulated_size = noiseFree1sppAccumulatedBufferDesc.byte_size;
    buffers.noisefree_1spp_accumulated[0].init(c, CL_MEM_READ_WRITE, noisefree_1spp_accumulated_size);
    buffers.noisefree_1spp_accumulated[1].init(c, CL_MEM_READ_WRITE, noisefree_1spp_accumulated_size);
	openCLBuffersTotalSize += 2 * noisefree_1spp_accumulated_size;

	// Final antialiased color buffer (3 * float32)
	const BufferDesc resultBufferDesc = GetResultBufferDesc(w, h);
	const size_t result_buffer_size = resultBufferDesc.byte_size;
    buffers.result_buffer[0].init(c, CL_MEM_READ_WRITE, result_buffer_size);
    buffers.result_buffer[1].init(c, CL_MEM_READ_WRITE, result_buffer_size);
	openCLBuffersTotalSize += 2 * result_buffer_size;

	// Previous frame pixel coordinates (after reprojection) (2 * float32)
	const BufferDesc prevFramePixelCoordsBufferDesc = GetPrevFramePixelCoordsBufferDesc(w, h);
	const size_t prev_frame_pixel_coords_buffer_size = prevFramePixelCoordsBufferDesc.byte_size;
    buffers.prev_frame_pixel_coords_buffer.init(c, CL_MEM_READ_WRITE, prev_frame_pixel_coords_buffer_size);
	openCLBuffersTotalSize += prev_frame_pixel_coords_buffer_size;

	// Validity mask of reprojected bilinear samples into previous frame (uchar 8bits)
	const BufferDesc prevFrameBilinearSamplesValidityMaskBufferDesc = GetPrevFrameBilinearSamplesValidityMaskBufferDesc(w, h);
	const size_t prev_frame_bilinear_samples_validity_mask_size = prevFrameBilinearSamplesValidityMaskBufferDesc.byte_size;
    buffers.prev_frame_bilinear_samples_validity_mask.init(c, CL_MEM_READ_WRITE, prev_frame_bilinear_samples_validity_mask_size);
	openCLBuffersTotalSize += prev_frame_bilinear_samples_validity_mask_size;

	// Tonemapped noise-free color estimate w/ albedo (3 * float32)
	const BufferDesc noiseFree1sppAccTonemappedBufferDesc = GetNoiseFree1sppAccTonemappedBufferDesc(w, h);
	const size_t noisefree_1spp_acc_tonemapped_size = noiseFree1sppAccTonemappedBufferDesc.byte_size;
    buffers.noisefree_1spp_acc_tonemapped.init(c, CL_MEM_READ_WRITE, noisefree_1spp_acc_tonemapped_size);
	openCLBuffersTotalSize += noisefree_1spp_acc_tonemapped_size;

	// Features weights per color channel (x3) (computed by the BMFR) (3 * float32)
	const size_t features_weights_buffer_size = WORKSET_WITH_MARGIN_BLOCK_COUNT * (features_count - 3) * 3 * sizeof(float);
    buffers.features_weights_buffer.init(c, CL_MEM_READ_WRITE, features_weights_buffer_size);
	openCLBuffersTotalSize += features_weights_buffer_size;

	// Min and max of features values per block (world_positions) (6 * 2 * float32)
	const size_t features_min_max_buffer_size = WORKSET_WITH_MARGIN_BLOCK_COUNT * 6 * 2 * sizeof(float);
    buffers.features_min_max_buffer.init(c, CL_MEM_READ_WRITE, features_min_max_buffer_size);
	openCLBuffersTotalSize += features_min_max_buffer_size;

	// Number of samples accumulated (for cumulative moving average) (char 8bits)
	const BufferDesc sppBufferDesc = GetSppBufferDesc(w, h);
	const size_t spp_buffer_size = sppBufferDesc.byte_size;
    buffers.spp_buffer[0].init(c, CL_MEM_READ_WRITE, spp_buffer_size);
    buffers.spp_buffer[1].init(c, CL_MEM_READ_WRITE, spp_buffer_size);
	openCLBuffersTotalSize += 2 * spp_buffer_size;

	// TODO: change this size to w * h
    buffers.result_host.resize(3 * OUTPUT_SIZE);

	LOG("OpenCL buffers total size: %.3fMB\n", float(openCLBuffersTotalSize) / 1024.f / 1024.f);
}

int bmfr_c_opencl(TmpData & tmpData)
{
    LOG("Initialize.\n");

	std::string features_not_scaled(NOT_SCALED_FEATURE_BUFFERS_STR);
	std::string features_scaled(SCALED_FEATURE_BUFFERS_STR);

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

	const size_t w = IMAGE_WIDTH;
	const size_t h = IMAGE_HEIGHT;

	const size_t localWidth					= GetLocalWidth();
	const size_t localHeight				= GetLocalHeight();
	const size_t worksetWidth				= ComputeWorksetWidth(w);
	const size_t worksetHeight				= ComputeWorksetHeight(h);
	const size_t worksetWidthWithMargin		= ComputeWorksetWidthWithMargin(w);
	const size_t worksetHeightWithMargin	= ComputeWorksetHeightWithMargin(h);
	const size_t fitterLocalSize			= GetFitterLocalSize();
	const size_t fitterGlobalSize			= GetFitterGlobalSize();


	// Create and build the kernel
	std::stringstream build_options;
	build_options <<
		" -D BUFFER_COUNT=" << buffer_count <<
		" -D FEATURES_NOT_SCALED=" << features_not_scaled_count <<
		" -D FEATURES_SCALED=" << features_scaled_count <<
		" -D IMAGE_WIDTH=" << w <<
		" -D IMAGE_HEIGHT=" << h <<
		" -D WORKSET_WIDTH=" << worksetWidth <<
		" -D WORKSET_HEIGHT=" << worksetHeight <<
		" -D FEATURE_BUFFERS=" << NOT_SCALED_FEATURE_BUFFERS_STR SCALED_FEATURE_BUFFERS_STR <<
		" -D LOCAL_WIDTH=" << localWidth <<
		" -D LOCAL_HEIGHT=" << localHeight <<
		" -D WORKSET_WITH_MARGINS_WIDTH=" << worksetWidthWithMargin <<
		" -D WORKSET_WITH_MARGINS_HEIGHT=" << worksetHeightWithMargin <<
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
		" -D LOCAL_SIZE=" << fitterLocalSize <<
		" -D USE_HALF_PRECISION_IN_FEATURES_DATA=" << STR(USE_HALF_PRECISION_IN_FEATURES_DATA);

	// Load the kernel source code into the array source_str
	FILE *fp;
	char *source_str;
	size_t source_size;
 
	fp = fopen(OPENCL_KERNEL_FILENAME, "r");
	if(!fp)
	{
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
    
	char * file_content = (char*)malloc(OPENCL_MAX_SOURCE_SIZE);

	source_str  = (char*)malloc(OPENCL_MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, OPENCL_MAX_SOURCE_SIZE, fp);
	fclose(fp);

	cl_int ret = CL_SUCCESS;
	cl_context context;
	cl_command_queue command_queue;
	cl_device_id device_id;
	init_opencl(context, command_queue, device_id);

	// Create a program from the kernel source
	cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);

	// Build the program
	if(clBuildProgram(program, 1, &device_id, build_options.str().c_str(), NULL, NULL) != CL_SUCCESS)
	{
		size_t len = 0;
		K_OPENCL_CHECK(clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &len));
		char *buffer = (char*)malloc(len * sizeof(char));
		K_OPENCL_CHECK(clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, len, buffer, NULL));

		LOG("Compilation failed: %s\n", buffer);
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


	// Create OpenCL buffers

	LOG("\nAllocate OpenCL buffers\n");
	
	BMFROpenCLBuffers buffers;
	init_bmfr_opencl_buffers(buffers, w, h, buffer_count, context);

	std::vector<Double_buffer<OpenCLDeviceBuffer> *> opencl_double_buffers =
	{
		&buffers.normals_buffer,
		&buffers.positions_buffer,
		&buffers.noisy_1spp_buffer,
		&buffers.noisefree_1spp_accumulated,
		&buffers.result_buffer,
		&buffers.spp_buffer
	};


	// Set kernel arguments
	int arg_index = 0;
	K_OPENCL_CHECK(clSetKernelArg(accum_noisy_kernel, arg_index++, sizeof(cl_mem), buffers.prev_frame_pixel_coords_buffer.data())); // [out] Previous frame pixel coordinates (after reprojection)
	K_OPENCL_CHECK(clSetKernelArg(accum_noisy_kernel, arg_index++, sizeof(cl_mem), buffers.prev_frame_bilinear_samples_validity_mask.data()));	// [out] Validity mask of bilinear samples in previous frame (after reprojection) (i.e valid reprojection = no disoclusion or outside frame)

	// https://www.khronos.org/registry/OpenCL/sdk/1.1/docs/man/xhtml/clSetKernelArg.html
	// Note: For arguments declared with the __local qualifier, the size specified will be the size in bytes of the buffer that must be allocated for the __local argument.
	arg_index = 0;
	K_OPENCL_CHECK(clSetKernelArg(fitter_kernel, arg_index++, LOCAL_SIZE * sizeof(float), nullptr));		// [local] Size of the shared memory used to perform parrallel reduction (max, min, sum)
	K_OPENCL_CHECK(clSetKernelArg(fitter_kernel, arg_index++, BLOCK_PIXELS * sizeof(float), nullptr));		// [local] Shared memory used to store the 'u' vectors
	K_OPENCL_CHECK(clSetKernelArg(fitter_kernel, arg_index++, r_size, nullptr));							// [local] Shared memory used to store the R matrix of the QR factorization
	K_OPENCL_CHECK(clSetKernelArg(fitter_kernel, arg_index++, sizeof(cl_mem), buffers.features_weights_buffer.data()));	// [out]   Features weights
	K_OPENCL_CHECK(clSetKernelArg(fitter_kernel, arg_index++, sizeof(cl_mem), buffers.features_min_max_buffer.data()));	// [out]   Min and max of features values per block (world_positions)

	arg_index = 0;
	K_OPENCL_CHECK(clSetKernelArg(weighted_sum_kernel, arg_index++, sizeof(cl_mem), buffers.features_weights_buffer.data()));	// [in]	 Features weights computed by the fitter kernel
	K_OPENCL_CHECK(clSetKernelArg(weighted_sum_kernel, arg_index++, sizeof(cl_mem), buffers.features_min_max_buffer.data()));	// [in]  Min and max of features values per block (world_positions)
	K_OPENCL_CHECK(clSetKernelArg(weighted_sum_kernel, arg_index++, sizeof(cl_mem), buffers.noisefree_1spp.data()));			// [out] Noise-free color estimate

	arg_index = 0;
	K_OPENCL_CHECK(clSetKernelArg(accum_filtered_kernel, arg_index++, sizeof(cl_mem), buffers.noisefree_1spp.data()));							// [in]  Noise free color estimate (computed as the weighted sum of the features)
	K_OPENCL_CHECK(clSetKernelArg(accum_filtered_kernel, arg_index++, sizeof(cl_mem), buffers.prev_frame_pixel_coords_buffer.data()));			// [in]  Previous frame pixel coordinates (after reprojection)
	K_OPENCL_CHECK(clSetKernelArg(accum_filtered_kernel, arg_index++, sizeof(cl_mem), buffers.prev_frame_bilinear_samples_validity_mask.data()));	// [in]  Validity mask of bilinear samples in previous frame (after reprojection)
	K_OPENCL_CHECK(clSetKernelArg(accum_filtered_kernel, arg_index++, sizeof(cl_mem), buffers.albedo_buffer.data()));								// [in]  Albedo buffer of the current frame (non-noisy)
	K_OPENCL_CHECK(clSetKernelArg(accum_filtered_kernel, arg_index++, sizeof(cl_mem), buffers.noisefree_1spp_acc_tonemapped.data()));				// [out] Accumulated and tonemapped noise-free color estimate

	arg_index = 0;
	K_OPENCL_CHECK(clSetKernelArg(taa_kernel, arg_index++, sizeof(cl_mem), buffers.prev_frame_pixel_coords_buffer.data()));	// [in] Previous frame pixel coordinates (after reprojection)
	K_OPENCL_CHECK(clSetKernelArg(taa_kernel, arg_index++, sizeof(cl_mem), buffers.noisefree_1spp_acc_tonemapped.data()));	// [in]	Current frame color buffer

	LOG("Run kernels.\n");

	// Note: enqueueNDRangeKernel takes in the global_size and the local_size.
	// In CUDA, a dispatch takes in the grid_size and the block_size.
	// The correspondance is as follow:
	//	global_size = grid_size * block_size
	//	local_size = block_size


	const size_t k_workset_with_margin_global_size[] = { worksetWidthWithMargin, worksetHeightWithMargin };
	const size_t k_workset_global_size[] = { worksetWidth, worksetHeight };
	const size_t k_local_size[] = { localWidth, localHeight };
	const size_t k_fitter_global_size[] = { fitterGlobalSize };
	const size_t k_fitter_local_size[] = { fitterLocalSize };

	FrameInputData frameInput;
	for(int frame = 0; frame < FRAME_COUNT; ++frame)
	{
		LOG("Frame %d\n", frame);

		LOG("  Load frame input data from disk\n");
		LoadFrameInputData(frameInput, w, h, frame);

		LOG("  Transfert data from host to device\n");
		const cl_bool blocking_write = true;
		K_OPENCL_CHECK(clEnqueueWriteBuffer(command_queue, *buffers.albedo_buffer.data(), blocking_write, 0, frameInput.albedos.size() * sizeof(float), frameInput.albedos.data(), 0, nullptr, nullptr));
		K_OPENCL_CHECK(clEnqueueWriteBuffer(command_queue, *buffers.normals_buffer.current().data(), blocking_write, 0, frameInput.normals.size() * sizeof(float), frameInput.normals.data(), 0, nullptr, nullptr));
		K_OPENCL_CHECK(clEnqueueWriteBuffer(command_queue, *buffers.positions_buffer.current().data(), blocking_write, 0, frameInput.positions.size() * sizeof(float), frameInput.positions.data(), 0, nullptr, nullptr));
		K_OPENCL_CHECK(clEnqueueWriteBuffer(command_queue, *buffers.noisy_1spp_buffer.current().data(), blocking_write, 0, frameInput.noisy1spps.size() * sizeof(float), frameInput.noisy1spps.data(), 0, nullptr, nullptr));

		// Phase I:
		//  - accumulate noisy 1spp
		//  - compute previous frame pixel coordinates (after reprojection)
		//  - generate validity bit mask of bilinear samples of previous frame
		//  - concatenate the different features in a single buffer
		// Note: On the first frame accum_noisy_kernel just copies to the features_buffer
		arg_index = 2;
		K_OPENCL_CHECK(clSetKernelArg(accum_noisy_kernel, arg_index++, sizeof(cl_mem), buffers.normals_buffer.current().data()));		// [in]  Current  (world) normals
		K_OPENCL_CHECK(clSetKernelArg(accum_noisy_kernel, arg_index++, sizeof(cl_mem), buffers.normals_buffer.previous().data()));		// [in]  Previous (world) normals
		K_OPENCL_CHECK(clSetKernelArg(accum_noisy_kernel, arg_index++, sizeof(cl_mem), buffers.positions_buffer.current().data()));		// [in]  Current  world positions
		K_OPENCL_CHECK(clSetKernelArg(accum_noisy_kernel, arg_index++, sizeof(cl_mem), buffers.positions_buffer.previous().data()));	// [in]  Previous world positions
		K_OPENCL_CHECK(clSetKernelArg(accum_noisy_kernel, arg_index++, sizeof(cl_mem), buffers.noisy_1spp_buffer.current().data()));	// [out] Current  noisy 1spp color
		K_OPENCL_CHECK(clSetKernelArg(accum_noisy_kernel, arg_index++, sizeof(cl_mem), buffers.noisy_1spp_buffer.previous().data()));	// [in]  Previous noisy 1spp color
		K_OPENCL_CHECK(clSetKernelArg(accum_noisy_kernel, arg_index++, sizeof(cl_mem), buffers.spp_buffer.previous().data()));			// [in]  Previous number of samples accumulated (for CMA)
		K_OPENCL_CHECK(clSetKernelArg(accum_noisy_kernel, arg_index++, sizeof(cl_mem), buffers.spp_buffer.current().data()));			// [out] Current  number of samples accumulated (for CMA)
		K_OPENCL_CHECK(clSetKernelArg(accum_noisy_kernel, arg_index++, sizeof(cl_mem), buffers.features_buffer.data()));				// [out] Features buffer (half or single-precision)
		const int matrix_index = frame == 0 ? 0 : frame - 1;
		K_OPENCL_CHECK(clSetKernelArg(accum_noisy_kernel, arg_index++, sizeof(cl_float16), &(camera_matrices[matrix_index][0][0]))); // [in] ViewProj matrix of previous frame
		K_OPENCL_CHECK(clSetKernelArg(accum_noisy_kernel, arg_index++, sizeof(cl_float2), &(pixel_offsets[frame][0])));
		K_OPENCL_CHECK(clSetKernelArg(accum_noisy_kernel, arg_index++, sizeof(cl_int), &frame)); // [in] Current frame number
		K_OPENCL_CHECK(clEnqueueNDRangeKernel(command_queue, accum_noisy_kernel, 2, NULL, k_workset_with_margin_global_size, k_local_size, 0, NULL, NULL));


		#if ENABLE_DEBUG_OUTPUT_TMP_DATA
		if(frame == DEBUG_OUTPUT_FRAME_NUMBER)
		{
			tmpData.features_buffer0.resize(buffers.features_buffer.size() / sizeof(float));
			K_OPENCL_CHECK(clEnqueueReadBuffer(command_queue, *buffers.features_buffer.data(), false, 0, buffers.features_buffer.size(), tmpData.features_buffer0.data(), 0, NULL, NULL));
		}
		#endif

		#if SAVE_INTERMEDIARY_BUFFERS
		SaveDevice3Float32ImageToDisk("noisy_1spp_buffer", frame, command_queue, *buffers.noisy_1spp_buffer.current().data(), GetNoisy1sppBufferDesc(w, h));
		#endif
 
		// Phase II: Blockwise Multi-Order Feature Regression (BMFR)
		// -> compute features weightss
		arg_index = 5;
		K_OPENCL_CHECK(clSetKernelArg(fitter_kernel, arg_index++, sizeof(cl_mem), buffers.features_buffer.data())); // [in] Features buffer (half or single-precision)
		K_OPENCL_CHECK(clSetKernelArg(fitter_kernel, arg_index++, sizeof(cl_int), &frame));  // [in] Current frame number
		K_OPENCL_CHECK(clEnqueueNDRangeKernel(command_queue, fitter_kernel, 1, NULL, k_fitter_global_size, k_fitter_local_size, 0, NULL, NULL));

		// Phase II: Compute noise free color estimate (weighted sum of features)
		arg_index = 3;
		K_OPENCL_CHECK(clSetKernelArg(weighted_sum_kernel, arg_index++, sizeof(cl_mem), buffers.normals_buffer.current().data()));		// [in] Current (world) normals
		K_OPENCL_CHECK(clSetKernelArg(weighted_sum_kernel, arg_index++, sizeof(cl_mem), buffers.positions_buffer.current().data()));	// [in] Current world positions
		K_OPENCL_CHECK(clSetKernelArg(weighted_sum_kernel, arg_index++, sizeof(cl_mem), buffers.noisy_1spp_buffer.current().data()));	// [in] Current noisy 1spp color (only used for debugging)
		K_OPENCL_CHECK(clSetKernelArg(weighted_sum_kernel, arg_index++, sizeof(cl_int), &frame));										// [in] Current frame number
		K_OPENCL_CHECK(clEnqueueNDRangeKernel(command_queue, weighted_sum_kernel, 2, NULL, k_workset_global_size, k_local_size, 0, NULL, NULL));

		#if SAVE_INTERMEDIARY_BUFFERS
		SaveDevice3Float32ImageToDisk("noisefree_1spp", frame, command_queue, *buffers.noisefree_1spp.data(), GetNoiseFree1sppBufferDesc(w, h));
		#endif

		// Phase III: Postprocessing
		// -> accumulate noise-free color estimate + output a tonemapped version
		arg_index = 5;
		K_OPENCL_CHECK(clSetKernelArg(accum_filtered_kernel, arg_index++, sizeof(cl_mem), buffers.spp_buffer.current().data())); // [in] Current number of samples accumulated (for CMA)
		K_OPENCL_CHECK(clSetKernelArg(accum_filtered_kernel, arg_index++, sizeof(cl_mem), buffers.noisefree_1spp_accumulated.previous().data())); // [in]  Previous frame noise-free accumulated color estimate 
		K_OPENCL_CHECK(clSetKernelArg(accum_filtered_kernel, arg_index++, sizeof(cl_mem), buffers.noisefree_1spp_accumulated.current().data()));  // [out] Current frame noise-free accumulated color estimate
		K_OPENCL_CHECK(clSetKernelArg(accum_filtered_kernel, arg_index++, sizeof(cl_int), &frame));	// [in] Current frame number
		K_OPENCL_CHECK(clEnqueueNDRangeKernel(command_queue, accum_filtered_kernel, 2, NULL, k_workset_global_size, k_local_size, 0, NULL, NULL));

		#if SAVE_INTERMEDIARY_BUFFERS
		SaveDevice3Float32ImageToDisk("noisefree_1spp_accumulated", frame, command_queue, *buffers.noisefree_1spp_accumulated.current().data(), GetNoiseFree1sppAccumulatedBufferDesc(w, h));
		#endif

		// Phase III: Temporal antialiasing
		arg_index = 2;
		K_OPENCL_CHECK(clSetKernelArg(taa_kernel, arg_index++, sizeof(cl_mem), buffers.result_buffer.current().data()));	// [out] Antialiased frame color buffer
		K_OPENCL_CHECK(clSetKernelArg(taa_kernel, arg_index++, sizeof(cl_mem), buffers.result_buffer.previous().data()));	// [in]  Previous frame color buffer
		K_OPENCL_CHECK(clSetKernelArg(taa_kernel, arg_index++, sizeof(cl_int), &frame));	// [in]  Current frame number
		K_OPENCL_CHECK(clEnqueueNDRangeKernel(command_queue, taa_kernel, 2, NULL, k_workset_global_size, k_local_size, 0, NULL, NULL));

		assert(ret == CL_SUCCESS);

		#if ENABLE_DEBUG_OUTPUT_TMP_DATA
		if(frame == DEBUG_OUTPUT_FRAME_NUMBER)
		{
			const size_t normals_buffer_size = buffers.normals_buffer.current().size();
			tmpData.normals.resize(normals_buffer_size / sizeof(float));
			K_OPENCL_CHECK(clEnqueueReadBuffer(command_queue, *buffers.normals_buffer.current().data(), false, 0, normals_buffer_size, tmpData.normals.data(), 0, NULL, NULL));

			const size_t positions_buffer_size = buffers.positions_buffer.current().size();
			tmpData.positions.resize(positions_buffer_size / sizeof(float));
			K_OPENCL_CHECK(clEnqueueReadBuffer(command_queue, *buffers.positions_buffer.current().data(), false, 0, positions_buffer_size, tmpData.positions.data(), 0, NULL, NULL));

			const size_t noisy_1spp_buffer_size = buffers.noisy_1spp_buffer.current().size();
			tmpData.noisy_1spp.resize(noisy_1spp_buffer_size / sizeof(float));
			K_OPENCL_CHECK(clEnqueueReadBuffer(command_queue, *buffers.noisy_1spp_buffer.current().data(), false, 0, noisy_1spp_buffer_size, tmpData.noisy_1spp.data(), 0, NULL, NULL));

			const size_t prev_frame_pixel_coords_buffer_size = buffers.prev_frame_pixel_coords_buffer.size();
			tmpData.prev_frame_pixel_coords_buffer.resize(prev_frame_pixel_coords_buffer_size / sizeof(float));
			K_OPENCL_CHECK(clEnqueueReadBuffer(command_queue, *buffers.prev_frame_pixel_coords_buffer.data(), false, 0, prev_frame_pixel_coords_buffer_size, tmpData.prev_frame_pixel_coords_buffer.data(), 0, NULL, NULL));

			const size_t prev_frame_bilinear_samples_validity_mask_size = buffers.prev_frame_bilinear_samples_validity_mask.size();
			tmpData.prev_frame_bilinear_samples_validity_mask.resize(prev_frame_bilinear_samples_validity_mask_size / sizeof(unsigned char));
			K_OPENCL_CHECK(clEnqueueReadBuffer(command_queue, *buffers.prev_frame_bilinear_samples_validity_mask.data(), false, 0, prev_frame_bilinear_samples_validity_mask_size, tmpData.prev_frame_bilinear_samples_validity_mask.data(), 0, NULL, NULL));

			const size_t spp_buffer_size = buffers.spp_buffer.current().size();
			tmpData.spp.resize(spp_buffer_size / sizeof(unsigned char));
			K_OPENCL_CHECK(clEnqueueReadBuffer(command_queue, *buffers.spp_buffer.current().data(), false, 0, spp_buffer_size, tmpData.spp.data(), 0, NULL, NULL));

			const size_t features_buffer_size = buffers.features_buffer.size();
			tmpData.features_buffer1.resize(features_buffer_size / sizeof(float));
			K_OPENCL_CHECK(clEnqueueReadBuffer(command_queue, *buffers.features_buffer.data(), false, 0, features_buffer_size, tmpData.features_buffer1.data(), 0, NULL, NULL));

			const size_t features_weights_buffer_size = buffers.features_weights_buffer.size();
			tmpData.features_weights_buffer.resize(features_weights_buffer_size / sizeof(float));
			K_OPENCL_CHECK(clEnqueueReadBuffer(command_queue, *buffers.features_weights_buffer.data(), false, 0, features_weights_buffer_size, tmpData.features_weights_buffer.data(), 0, NULL, NULL));

			const size_t features_min_max_buffer_size = buffers.features_min_max_buffer.size();
			tmpData.features_min_max_buffer.resize(features_min_max_buffer_size / sizeof(float));
			K_OPENCL_CHECK(clEnqueueReadBuffer(command_queue, *buffers.features_min_max_buffer.data(), false, 0, features_min_max_buffer_size, tmpData.features_min_max_buffer.data(), 0, NULL, NULL));

			const size_t noisefree_1spp_size = buffers.noisefree_1spp.size();
			tmpData.noisefree_1spp.resize(noisefree_1spp_size / sizeof(float));
			K_OPENCL_CHECK(clEnqueueReadBuffer(command_queue, *buffers.noisefree_1spp.data(), false, 0, noisefree_1spp_size, tmpData.noisefree_1spp.data(), 0, NULL, NULL));

			const size_t noisefree_1spp_accumulated_size = buffers.noisefree_1spp_accumulated.current().size();
			tmpData.noisefree_1spp_accumulated.resize(noisefree_1spp_accumulated_size / sizeof(float));
			K_OPENCL_CHECK(clEnqueueReadBuffer(command_queue, *buffers.noisefree_1spp_accumulated.current().data(), false, 0, noisefree_1spp_accumulated_size, tmpData.noisefree_1spp_accumulated.data(), 0, NULL, NULL));

			const size_t noisefree_1spp_acc_tonemapped_size = buffers.noisefree_1spp_acc_tonemapped.size();
			tmpData.noisefree_1spp_acc_tonemapped.resize(noisefree_1spp_acc_tonemapped_size / sizeof(float));
			K_OPENCL_CHECK(clEnqueueReadBuffer(command_queue, *buffers.noisefree_1spp_acc_tonemapped.data(), false, 0, noisefree_1spp_acc_tonemapped_size, tmpData.noisefree_1spp_acc_tonemapped.data(), 0, NULL, NULL));

			const size_t result_buffer_size = buffers.result_buffer.current().size();
			tmpData.result.resize(result_buffer_size / sizeof(float));
			K_OPENCL_CHECK(clEnqueueReadBuffer(command_queue, *buffers.result_buffer.current().data(), false, 0, result_buffer_size, tmpData.result.data(), 0, NULL, NULL));

			K_OPENCL_CHECK(clFlush(command_queue));
			K_OPENCL_CHECK(clFinish(command_queue));

			return 0;
		}
		#endif

		SaveDevice3Float32ImageToDisk("result", frame, command_queue, *buffers.result_buffer.current().data(), GetResultBufferDesc(w, h));

		// Swap all double buffers
		std::for_each(opencl_double_buffers.begin(), opencl_double_buffers.end(), std::bind(&Double_buffer<OpenCLDeviceBuffer>::swap, std::placeholders::_1));
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
	K_OPENCL_CHECK(clReleaseCommandQueue(command_queue));
	K_OPENCL_CHECK(clReleaseContext(context));

	return 0;
}
