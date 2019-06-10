#include "bmfr_cuda_opencl.hpp"
#include "bmfr_cuda.hpp"
#include "bmfr_c_opencl.hpp"
#include "utils.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <functional>
#include <CL/cl.h>

#include "bmfr.cuh"

void CopyOpenCLBufferToCudaBuffer(OpenCLDeviceBuffer const & src, CudaDeviceBuffer & dst, cl_command_queue & command_queue)
{
	const size_t datasize = dst.size();
	std::vector<char> hostBuffer;
	hostBuffer.resize(datasize);

	K_OPENCL_CHECK(clEnqueueReadBuffer(command_queue, *src.data(), false, 0, datasize, hostBuffer.data(), 0, NULL, NULL));
	K_OPENCL_CHECK(clFlush(command_queue));
	K_OPENCL_CHECK(clFinish(command_queue));
	K_CUDA_CHECK(cudaMemcpy(dst.data(), hostBuffer.data(), datasize, cudaMemcpyHostToDevice));
	K_CUDA_CHECK(cudaDeviceSynchronize());
}

void CompareOpenCLBufferAndCudaBuffer(char const * name, OpenCLDeviceBuffer const & clBuffer, CudaDeviceBuffer const & cuBuffer, cl_command_queue & command_queue)
{
	assert(clBuffer.size() == cuBuffer.size());

	const size_t buffer_size = clBuffer.size();

	// OpenCL
	std::vector<float> clBufferHost;
	clBufferHost.resize(buffer_size / sizeof(float));
	K_OPENCL_CHECK(clEnqueueReadBuffer(command_queue, *clBuffer.data(), false, 0, buffer_size, clBufferHost.data(), 0, NULL, NULL));
	K_OPENCL_CHECK(clFlush(command_queue));
	K_OPENCL_CHECK(clFinish(command_queue));

	// Cuda
	std::vector<float> cuBufferHost;
	cuBufferHost.resize(buffer_size / sizeof(float));
	K_CUDA_CHECK(cudaMemcpy(cuBufferHost.data(), cuBuffer.data(), buffer_size, cudaMemcpyDeviceToHost));
	K_CUDA_CHECK(cudaDeviceSynchronize());

	CheckDiffFloat(name, cuBufferHost, clBufferHost);

	printf("%s\n", name);
}

void ClearBuffer(OpenCLDeviceBuffer & clBuffer, cl_command_queue & command_queue, char value = 0)
{
	const size_t buffer_size = clBuffer.size();
	std::vector<char> clBufferHost;
	clBufferHost.resize(buffer_size);
	for(size_t i = 0; i < buffer_size; ++i) clBufferHost[i] = value;
	K_OPENCL_CHECK(clEnqueueWriteBuffer(command_queue, *clBuffer.data(), true, 0, buffer_size, clBufferHost.data(), 0, nullptr, nullptr));
	K_OPENCL_CHECK(clFlush(command_queue));
	K_OPENCL_CHECK(clFinish(command_queue));
}

void ClearBufferFloat(OpenCLDeviceBuffer & clBuffer, cl_command_queue & command_queue, float value = 0)
{
	const size_t buffer_size = clBuffer.size();
	std::vector<float> clBufferHost;
	clBufferHost.resize(buffer_size / sizeof(float));
	for(size_t i = 0; i < clBufferHost.size(); ++i) clBufferHost[i] = value;
	K_OPENCL_CHECK(clEnqueueWriteBuffer(command_queue, *clBuffer.data(), true, 0, buffer_size, clBufferHost.data(), 0, nullptr, nullptr));
	K_OPENCL_CHECK(clFlush(command_queue));
	K_OPENCL_CHECK(clFinish(command_queue));
}

void ClearBuffer(CudaDeviceBuffer & cuBuffer, char value = 0)
{
	const size_t buffer_size = cuBuffer.size();
	std::vector<char> cuBufferHost;
	cuBufferHost.resize(buffer_size);
	for(size_t i = 0; i < buffer_size; ++i) cuBufferHost[i] = value;
	K_CUDA_CHECK(cudaMemcpy(cuBuffer.data(), cuBufferHost.data(), buffer_size, cudaMemcpyHostToDevice));
	K_CUDA_CHECK(cudaDeviceSynchronize());
}


void ClearBufferFloat(CudaDeviceBuffer & cuBuffer, float value = 0)
{
	const size_t buffer_size = cuBuffer.size();
	std::vector<float> cuBufferHost;
	cuBufferHost.resize(buffer_size / sizeof(float));
	for(size_t i = 0; i < cuBufferHost.size(); ++i) cuBufferHost[i] = value;
	K_CUDA_CHECK(cudaMemcpy(cuBuffer.data(), cuBufferHost.data(), buffer_size, cudaMemcpyHostToDevice));
	K_CUDA_CHECK(cudaDeviceSynchronize());
}

void bmfr_cuda_c_opencl(TmpData & cudaTmpData, TmpData & openclTmpData)
{
	// Common init /////////////////////////////////////////////////////////////

	LOG("Initialize.\n");

	std::string features_not_scaled(NOT_SCALED_FEATURE_BUFFERS_STR);
	std::string features_scaled(SCALED_FEATURE_BUFFERS_STR);

	const int features_not_scaled_count = FEATURES_NOT_SCALED;
	const int features_scaled_count = FEATURES_SCALED;
	
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
	const size_t worksetWithMarginWidth		= ComputeWorksetWithMarginWidth(w);
	const size_t worksetWithMarginHeight	= ComputeWorksetWithMarginHeight(h);
	const size_t fitterLocalSize			= GetFitterLocalSize();
	const size_t fitterGlobalSize			= GetFitterGlobalSize(w, h);


	// OpenCL init /////////////////////////////////////////////////////////////

	// Create and build the kernel
	std::stringstream build_options;
	build_options <<
		" -cl-opt-disable" <<
		//" -cl-denorms-are-zero" <<
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
		" -D WORKSET_WITH_MARGINS_WIDTH=" << worksetWithMarginWidth <<
		" -D WORKSET_WITH_MARGINS_HEIGHT=" << worksetWithMarginHeight <<
		" -D BLOCK_EDGE_LENGTH=" << STR(BLOCK_EDGE_LENGTH) <<
		" -D BLOCK_PIXELS=" << BLOCK_PIXELS <<
		" -D R_EDGE=" << buffer_count - 2 <<
		" -D NOISE_AMOUNT=" << STR(NOISE_AMOUNT) <<
		" -D BLEND_ALPHA=" << STR(BLEND_ALPHA) <<
		" -D SECOND_BLEND_ALPHA=" << STR(SECOND_BLEND_ALPHA) <<
		" -D TAA_BLEND_ALPHA=" << STR(TAA_BLEND_ALPHA) <<

		// TODO: use the values from the scene files
#if 0
		" -D POSITION_LIMIT_SQUARED=" << position_limit_squared <<
		" -D NORMAL_LIMIT_SQUARED=" << normal_limit_squared <<
#else
		" -D POSITION_LIMIT_SQUARED=" << POSITION_LIMIT_SQUARED <<
		" -D NORMAL_LIMIT_SQUARED=" << NORMAL_LIMIT_SQUARED <<
#endif
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
	
	BMFROpenCLBuffers cl_buffers;
	init_bmfr_opencl_buffers(cl_buffers, w, h, buffer_count, context);

	std::vector<Double_buffer<OpenCLDeviceBuffer> *> opencl_double_buffers =
	{
		&cl_buffers.normals_buffer,
		&cl_buffers.positions_buffer,
		&cl_buffers.noisy_1spp_buffer,
		&cl_buffers.noisefree_1spp_accumulated,
		&cl_buffers.result_buffer,
		&cl_buffers.spp_buffer
	};


	// Set kernel arguments
	int arg_index = 0;
	K_OPENCL_CHECK(clSetKernelArg(accum_noisy_kernel, arg_index++, sizeof(cl_mem), cl_buffers.prev_frame_pixel_coords_buffer.data())); // [out] Previous frame pixel coordinates (after reprojection)
	K_OPENCL_CHECK(clSetKernelArg(accum_noisy_kernel, arg_index++, sizeof(cl_mem), cl_buffers.prev_frame_bilinear_samples_validity_mask.data()));	// [out] Validity mask of bilinear samples in previous frame (after reprojection) (i.e valid reprojection = no disoclusion or outside frame)

	// https://www.khronos.org/registry/OpenCL/sdk/1.1/docs/man/xhtml/clSetKernelArg.html
	// Note: For arguments declared with the __local qualifier, the size specified will be the size in bytes of the buffer that must be allocated for the __local argument.
	arg_index = 0;
	K_OPENCL_CHECK(clSetKernelArg(fitter_kernel, arg_index++, LOCAL_SIZE * sizeof(float), nullptr));		// [local] Size of the shared memory used to perform parrallel reduction (max, min, sum)
	K_OPENCL_CHECK(clSetKernelArg(fitter_kernel, arg_index++, BLOCK_PIXELS * sizeof(float), nullptr));		// [local] Shared memory used to store the 'u' vectors
	K_OPENCL_CHECK(clSetKernelArg(fitter_kernel, arg_index++, r_size, nullptr));							// [local] Shared memory used to store the R matrix of the QR factorization
	K_OPENCL_CHECK(clSetKernelArg(fitter_kernel, arg_index++, sizeof(cl_mem), cl_buffers.features_weights_buffer.data()));	// [out]   Features weights
	K_OPENCL_CHECK(clSetKernelArg(fitter_kernel, arg_index++, sizeof(cl_mem), cl_buffers.features_min_max_buffer.data()));	// [out]   Min and max of features values per block (world_positions)

	arg_index = 0;
	K_OPENCL_CHECK(clSetKernelArg(weighted_sum_kernel, arg_index++, sizeof(cl_mem), cl_buffers.features_weights_buffer.data()));	// [in]	 Features weights computed by the fitter kernel
	K_OPENCL_CHECK(clSetKernelArg(weighted_sum_kernel, arg_index++, sizeof(cl_mem), cl_buffers.features_min_max_buffer.data()));	// [in]  Min and max of features values per block (world_positions)
	K_OPENCL_CHECK(clSetKernelArg(weighted_sum_kernel, arg_index++, sizeof(cl_mem), cl_buffers.noisefree_1spp.data()));			// [out] Noise-free color estimate

	arg_index = 0;
	K_OPENCL_CHECK(clSetKernelArg(accum_filtered_kernel, arg_index++, sizeof(cl_mem), cl_buffers.noisefree_1spp.data()));							// [in]  Noise free color estimate (computed as the weighted sum of the features)
	K_OPENCL_CHECK(clSetKernelArg(accum_filtered_kernel, arg_index++, sizeof(cl_mem), cl_buffers.prev_frame_pixel_coords_buffer.data()));			// [in]  Previous frame pixel coordinates (after reprojection)
	K_OPENCL_CHECK(clSetKernelArg(accum_filtered_kernel, arg_index++, sizeof(cl_mem), cl_buffers.prev_frame_bilinear_samples_validity_mask.data()));	// [in]  Validity mask of bilinear samples in previous frame (after reprojection)
	K_OPENCL_CHECK(clSetKernelArg(accum_filtered_kernel, arg_index++, sizeof(cl_mem), cl_buffers.albedo_buffer.data()));								// [in]  Albedo buffer of the current frame (non-noisy)
	K_OPENCL_CHECK(clSetKernelArg(accum_filtered_kernel, arg_index++, sizeof(cl_mem), cl_buffers.noisefree_1spp_acc_tonemapped.data()));				// [out] Accumulated and tonemapped noise-free color estimate

	arg_index = 0;
	K_OPENCL_CHECK(clSetKernelArg(taa_kernel, arg_index++, sizeof(cl_mem), cl_buffers.prev_frame_pixel_coords_buffer.data()));	// [in] Previous frame pixel coordinates (after reprojection)
	K_OPENCL_CHECK(clSetKernelArg(taa_kernel, arg_index++, sizeof(cl_mem), cl_buffers.noisefree_1spp_acc_tonemapped.data()));	// [in]	Current frame color buffer


	// Note: enqueueNDRangeKernel takes in the global_size and the local_size.
	// In CUDA, a dispatch takes in the grid_size and the block_size.
	// The correspondance is as follow:
	//	global_size = grid_size * block_size
	//	local_size = block_size

	const size_t k_workset_with_margin_global_size[] = { worksetWithMarginWidth, worksetWithMarginHeight };
	const size_t k_workset_global_size[] = { worksetWidth, worksetHeight };
	const size_t k_local_size[] = { localWidth, localHeight };
	const size_t k_fitter_global_size[] = { fitterGlobalSize };
	const size_t k_fitter_local_size[] = { fitterLocalSize };


	// CUDA init ///////////////////////////////////////////////////////////////

	// Create CUDA buffers

	LOG("\nAllocate CUDA buffers\n");

	BMFRCudaBuffers cu_buffers;
	init_bmfr_cuda_buffers(cu_buffers, w, h, buffer_count);

	std::vector<Double_buffer<CudaDeviceBuffer> *> cuda_double_buffers =
	{
		&cu_buffers.normals_buffer,
		&cu_buffers.positions_buffer,
		&cu_buffers.noisy_1spp_buffer,
		&cu_buffers.noisefree_1spp_accumulated,
		&cu_buffers.result_buffer,
		&cu_buffers.spp_buffer
	};

    const dim3 k_block_size(localWidth, localHeight);
    const dim3 k_workset_grid_size((worksetWidth + k_block_size.x - 1) / k_block_size.x, (worksetHeight + k_block_size.y - 1) / k_block_size.y);
	const dim3 k_workset_with_margin_grid_size((worksetWithMarginWidth + k_block_size.x - 1) / k_block_size.x, (worksetWithMarginHeight + k_block_size.y - 1) / k_block_size.y);
    const dim3 k_fitter_block_size(fitterLocalSize);
    const dim3 k_fitter_grid_size((fitterGlobalSize + k_fitter_block_size.x - 1) / k_fitter_block_size.x);


	// Processing //////////////////////////////////////////////////////////////
	
	LOG("Run kernels.\n");

	FrameInputData frameInput;
	for(int frame = 0; frame < FRAME_COUNT; ++frame)
	{
		LOG("Frame %d\n", frame);

		LOG("  Load frame input data from disk\n");
		LoadFrameInputData(frameInput, w, h, frame);

		LOG("  Transfert data from host to device\n");
		const cl_bool blocking_write = true;
		K_OPENCL_CHECK(clEnqueueWriteBuffer(command_queue, *cl_buffers.albedo_buffer.data(), blocking_write, 0, frameInput.albedos.size() * sizeof(float), frameInput.albedos.data(), 0, nullptr, nullptr));
		K_OPENCL_CHECK(clEnqueueWriteBuffer(command_queue, *cl_buffers.normals_buffer.current().data(), blocking_write, 0, frameInput.normals.size() * sizeof(float), frameInput.normals.data(), 0, nullptr, nullptr));
		K_OPENCL_CHECK(clEnqueueWriteBuffer(command_queue, *cl_buffers.positions_buffer.current().data(), blocking_write, 0, frameInput.positions.size() * sizeof(float), frameInput.positions.data(), 0, nullptr, nullptr));
		K_OPENCL_CHECK(clEnqueueWriteBuffer(command_queue, *cl_buffers.frame_noisy_1spp_buffer.data(), blocking_write, 0, frameInput.noisy1spps.size() * sizeof(float), frameInput.noisy1spps.data(), 0, nullptr, nullptr));
		K_CUDA_CHECK(cudaMemcpy(cu_buffers.albedo_buffer.data(), frameInput.albedos.data(), frameInput.albedos.size() * sizeof(float), cudaMemcpyHostToDevice));
        K_CUDA_CHECK(cudaMemcpy(cu_buffers.normals_buffer.current().data(), frameInput.normals.data(), frameInput.normals.size() * sizeof(float), cudaMemcpyHostToDevice));
        K_CUDA_CHECK(cudaMemcpy(cu_buffers.positions_buffer.current().data(), frameInput.positions.data(), frameInput.positions.size() * sizeof(float), cudaMemcpyHostToDevice));
        K_CUDA_CHECK(cudaMemcpy(cu_buffers.frame_noisy_1spp_buffer.data(), frameInput.noisy1spps.data(), frameInput.noisy1spps.size() * sizeof(float), cudaMemcpyHostToDevice));

		// Phase I:
		//  - accumulate noisy 1spp
		//  - compute previous frame pixel coordinates (after reprojection)
		//  - generate validity bit mask of bilinear samples of previous frame
		//  - concatenate the different features in a single buffer
		// Note: On the first frame accum_noisy_kernel just copies to the features_buffer
		arg_index = 2;
		K_OPENCL_CHECK(clSetKernelArg(accum_noisy_kernel, arg_index++, sizeof(cl_mem), cl_buffers.normals_buffer.current().data()));		// [in]  Current  (world) normals
		K_OPENCL_CHECK(clSetKernelArg(accum_noisy_kernel, arg_index++, sizeof(cl_mem), cl_buffers.normals_buffer.previous().data()));		// [in]  Previous (world) normals
		K_OPENCL_CHECK(clSetKernelArg(accum_noisy_kernel, arg_index++, sizeof(cl_mem), cl_buffers.positions_buffer.current().data()));		// [in]  Current  world positions
		K_OPENCL_CHECK(clSetKernelArg(accum_noisy_kernel, arg_index++, sizeof(cl_mem), cl_buffers.positions_buffer.previous().data()));		// [in]  Previous world positions
		K_OPENCL_CHECK(clSetKernelArg(accum_noisy_kernel, arg_index++, sizeof(cl_mem), cl_buffers.frame_noisy_1spp_buffer.data()));			// [in]  Frame noisy 1spp color
		K_OPENCL_CHECK(clSetKernelArg(accum_noisy_kernel, arg_index++, sizeof(cl_mem), cl_buffers.noisy_1spp_buffer.current().data()));		// [out] Current  accumulated noisy 1spp color
		K_OPENCL_CHECK(clSetKernelArg(accum_noisy_kernel, arg_index++, sizeof(cl_mem), cl_buffers.noisy_1spp_buffer.previous().data()));	// [in]  Previous accumulated noisy 1spp color
		K_OPENCL_CHECK(clSetKernelArg(accum_noisy_kernel, arg_index++, sizeof(cl_mem), cl_buffers.spp_buffer.previous().data()));			// [in]  Previous number of samples accumulated (for CMA)
		K_OPENCL_CHECK(clSetKernelArg(accum_noisy_kernel, arg_index++, sizeof(cl_mem), cl_buffers.spp_buffer.current().data()));			// [out] Current  number of samples accumulated (for CMA)
		K_OPENCL_CHECK(clSetKernelArg(accum_noisy_kernel, arg_index++, sizeof(cl_mem), cl_buffers.features_buffer.data()));					// [out] Features buffer (half or single-precision)
		const int matrix_index = frame == 0 ? 0 : frame - 1;
		K_OPENCL_CHECK(clSetKernelArg(accum_noisy_kernel, arg_index++, sizeof(cl_float16), &(camera_matrices[matrix_index][0][0]))); // [in] ViewProj matrix of previous frame
		K_OPENCL_CHECK(clSetKernelArg(accum_noisy_kernel, arg_index++, sizeof(cl_float2), &(pixel_offsets[frame][0])));
		K_OPENCL_CHECK(clSetKernelArg(accum_noisy_kernel, arg_index++, sizeof(cl_int), &frame)); // [in] Current frame number
		K_OPENCL_CHECK(clEnqueueNDRangeKernel(command_queue, accum_noisy_kernel, 2, NULL, k_workset_with_margin_global_size, k_local_size, 0, NULL, NULL));

		const mat4x4 cam_mat = *reinterpret_cast<mat4x4 const *>(&camera_matrices[matrix_index][0][0]);
		const vec2 pix_off = *reinterpret_cast<vec2 const *>(&pixel_offsets[frame][0]);

		run_accumulate_noisy_data(
			k_workset_with_margin_grid_size,
			k_block_size,
			cu_buffers.prev_frame_pixel_coords_buffer.getTypedData<vec2>(),
			cu_buffers.prev_frame_bilinear_samples_validity_mask.getTypedData<unsigned char>(),
			cu_buffers.normals_buffer.current().getTypedData<float>(),
			cu_buffers.normals_buffer.previous().getTypedData<float>(),
			cu_buffers.positions_buffer.current().getTypedData<float>(),
			cu_buffers.positions_buffer.previous().getTypedData<float>(),
			cu_buffers.frame_noisy_1spp_buffer.getTypedData<float>(),
			cu_buffers.noisy_1spp_buffer.current().getTypedData<float>(),
			cu_buffers.noisy_1spp_buffer.previous().getTypedData<float>(),
			// TODO: invert the order of the spp buffers
			cu_buffers.spp_buffer.previous().getTypedData<unsigned char>(),
			cu_buffers.spp_buffer.current().getTypedData<unsigned char>(),
			#if USE_HALF_PRECISION_IN_FEATURES_DATA
			cu_buffers.features_buffer.getTypedData<half>(),
			#else
			cu_buffers.features_buffer.getTypedData<float>(),
			#endif
			cam_mat,
			pix_off,
			frame
		);

		// Check results against reference
		//CompareOpenCLBufferAndCudaBuffer("spp_buffer", cl_buffers.spp_buffer.current(), cu_buffers.spp_buffer.current(), command_queue);
		//CompareOpenCLBufferAndCudaBuffer("features_buffer", cl_buffers.features_buffer, cu_buffers.features_buffer, command_queue);
		//CompareOpenCLBufferAndCudaBuffer("prev_frame_pixel_coords_buffer", cl_buffers.prev_frame_pixel_coords_buffer, cu_buffers.prev_frame_pixel_coords_buffer, command_queue);
		//CompareOpenCLBufferAndCudaBuffer("prev_frame_bilinear_samples_validity_mask", cl_buffers.prev_frame_bilinear_samples_validity_mask, cu_buffers.prev_frame_bilinear_samples_validity_mask, command_queue);
		//CompareOpenCLBufferAndCudaBuffer("noisy_1spp_buffer", cl_buffers.noisy_1spp_buffer.current(), cu_buffers.noisy_1spp_buffer.current(), command_queue);
		//CompareOpenCLBufferAndCudaBuffer("noisy_1spp_buffer_prev", cl_buffers.noisy_1spp_buffer.previous(), cu_buffers.noisy_1spp_buffer.previous(), command_queue);

		//CopyOpenCLBufferToCudaBuffer(cl_buffers.spp_buffer.current(), cu_buffers.spp_buffer.current(), command_queue);
		//CopyOpenCLBufferToCudaBuffer(cl_buffers.features_buffer, cu_buffers.features_buffer, command_queue);
		//CopyOpenCLBufferToCudaBuffer(cl_buffers.prev_frame_pixel_coords_buffer, cu_buffers.prev_frame_pixel_coords_buffer, command_queue);
		//CopyOpenCLBufferToCudaBuffer(cl_buffers.prev_frame_bilinear_samples_validity_mask, cu_buffers.prev_frame_bilinear_samples_validity_mask, command_queue);
		//CopyOpenCLBufferToCudaBuffer(cl_buffers.noisy_1spp_buffer.current(), cu_buffers.noisy_1spp_buffer.current(), command_queue);

		#if ENABLE_DEBUG_OUTPUT_TMP_DATA
		if(frame == DEBUG_OUTPUT_FRAME_NUMBER)
		{
			openclTmpData.features_buffer0.resize(cl_buffers.features_buffer.size() / sizeof(float));
			K_OPENCL_CHECK(clEnqueueReadBuffer(command_queue, *cl_buffers.features_buffer.data(), false, 0, cl_buffers.features_buffer.size(), openclTmpData.features_buffer0.data(), 0, NULL, NULL));
		}
		#endif

		#if ENABLE_DEBUG_OUTPUT_TMP_DATA
		if(frame == DEBUG_OUTPUT_FRAME_NUMBER)
		{
			cudaTmpData.features_buffer0.resize(cu_buffers.features_buffer.size() / sizeof(float));
			K_CUDA_CHECK(cudaMemcpy(cudaTmpData.features_buffer0.data(), cu_buffers.features_buffer.data(), cu_buffers.features_buffer.size(), cudaMemcpyDeviceToHost));
		}
		#endif

		#if SAVE_INTERMEDIARY_BUFFERS
		SaveDevice3Float32ImageToDisk("noisy_1spp_buffer", frame, command_queue, *cl_buffers.noisy_1spp_buffer.current().data(), GetNoisy1sppBufferDesc(w, h));
		SaveDevice3Float32ImageToDisk("noisy_1spp_buffer", frame, cu_buffers.noisy_1spp_buffer.current(), GetNoisy1sppBufferDesc(w, h));
		#endif
 
		// Phase II: Blockwise Multi-Order Feature Regression (BMFR)
		// -> compute features weightss
		arg_index = 5;
		K_OPENCL_CHECK(clSetKernelArg(fitter_kernel, arg_index++, sizeof(cl_mem), cl_buffers.features_buffer.data())); // [in] Features buffer (half or single-precision)
		K_OPENCL_CHECK(clSetKernelArg(fitter_kernel, arg_index++, sizeof(cl_int), &frame));  // [in] Current frame number
		K_OPENCL_CHECK(clEnqueueNDRangeKernel(command_queue, fitter_kernel, 1, NULL, k_fitter_global_size, k_fitter_local_size, 0, NULL, NULL));

		run_fitter(
			k_fitter_grid_size,
			k_fitter_block_size,
			cu_buffers.features_weights_buffer.getTypedData<float>(),
			cu_buffers.features_min_max_buffer.getTypedData<float>(),
			#if USE_HALF_PRECISION_IN_FEATURES_DATA
			cu_buffers.features_buffer.getTypedData<half>(),
			#else
			cu_buffers.features_buffer.getTypedData<float>(),
			#endif
			frame
		);

		// Check results against reference
		//CompareOpenCLBufferAndCudaBuffer("features_weights_buffer", cl_buffers.features_weights_buffer, cu_buffers.features_weights_buffer, command_queue);

		// Identical!
		//CompareOpenCLBufferAndCudaBuffer("features_min_max_buffer", cl_buffers.features_min_max_buffer, cu_buffers.features_min_max_buffer, command_queue);

		//CompareOpenCLBufferAndCudaBuffer("features_buffer", cl_buffers.features_buffer, cu_buffers.features_buffer, command_queue);

		//CopyOpenCLBufferToCudaBuffer(cl_buffers.features_weights_buffer, cu_buffers.features_weights_buffer, command_queue);
		//CopyOpenCLBufferToCudaBuffer(cl_buffers.features_min_max_buffer, cu_buffers.features_min_max_buffer, command_queue);
		//CopyOpenCLBufferToCudaBuffer(cl_buffers.features_buffer, cu_buffers.features_buffer, command_queue);

		// Phase II: Compute noise free color estimate (weighted sum of features)
		arg_index = 3;
		K_OPENCL_CHECK(clSetKernelArg(weighted_sum_kernel, arg_index++, sizeof(cl_mem), cl_buffers.normals_buffer.current().data()));		// [in] Current (world) normals
		K_OPENCL_CHECK(clSetKernelArg(weighted_sum_kernel, arg_index++, sizeof(cl_mem), cl_buffers.positions_buffer.current().data()));	// [in] Current world positions
		K_OPENCL_CHECK(clSetKernelArg(weighted_sum_kernel, arg_index++, sizeof(cl_mem), cl_buffers.noisy_1spp_buffer.current().data()));	// [in] Current noisy 1spp color (only used for debugging)
		K_OPENCL_CHECK(clSetKernelArg(weighted_sum_kernel, arg_index++, sizeof(cl_int), &frame));										// [in] Current frame number
		K_OPENCL_CHECK(clEnqueueNDRangeKernel(command_queue, weighted_sum_kernel, 2, NULL, k_workset_global_size, k_local_size, 0, NULL, NULL));

		run_weighted_sum(
			k_workset_grid_size,
			k_block_size,
			cu_buffers.features_weights_buffer.getTypedData<float>(),
			cu_buffers.features_min_max_buffer.getTypedData<float>(),
			cu_buffers.noisefree_1spp.getTypedData<float>(),
			cu_buffers.normals_buffer.current().getTypedData<float>(),
			cu_buffers.positions_buffer.current().getTypedData<float>(),
			cu_buffers.noisy_1spp_buffer.current().getTypedData<float>(),
			frame
		);

		// Check results against reference
		//CompareOpenCLBufferAndCudaBuffer("noisefree_1spp", cl_buffers.noisefree_1spp, cu_buffers.noisefree_1spp, command_queue);

		//CopyOpenCLBufferToCudaBuffer(cl_buffers.noisefree_1spp, cu_buffers.noisefree_1spp, command_queue);

		#if SAVE_INTERMEDIARY_BUFFERS
		SaveDevice3Float32ImageToDisk("noisefree_1spp", frame, command_queue, *cl_buffers.noisefree_1spp.data(), GetNoiseFree1sppBufferDesc(w, h));
		SaveDevice3Float32ImageToDisk("noisefree_1spp", frame, cu_buffers.noisefree_1spp, GetNoiseFree1sppBufferDesc(w, h));
		#endif

		// Phase III: Postprocessing
		// -> accumulate noise-free color estimate + output a tonemapped version
		arg_index = 5;
		K_OPENCL_CHECK(clSetKernelArg(accum_filtered_kernel, arg_index++, sizeof(cl_mem), cl_buffers.spp_buffer.current().data())); // [in] Current number of samples accumulated (for CMA)
		K_OPENCL_CHECK(clSetKernelArg(accum_filtered_kernel, arg_index++, sizeof(cl_mem), cl_buffers.noisefree_1spp_accumulated.previous().data())); // [in]  Previous frame noise-free accumulated color estimate 
		K_OPENCL_CHECK(clSetKernelArg(accum_filtered_kernel, arg_index++, sizeof(cl_mem), cl_buffers.noisefree_1spp_accumulated.current().data()));  // [out] Current frame noise-free accumulated color estimate
		K_OPENCL_CHECK(clSetKernelArg(accum_filtered_kernel, arg_index++, sizeof(cl_int), &frame));	// [in] Current frame number
		K_OPENCL_CHECK(clEnqueueNDRangeKernel(command_queue, accum_filtered_kernel, 2, NULL, k_workset_global_size, k_local_size, 0, NULL, NULL));

		run_accumulate_filtered_data(
			k_workset_grid_size,
			k_block_size,
			cu_buffers.noisefree_1spp.getTypedData<float>(),
			cu_buffers.prev_frame_pixel_coords_buffer.getTypedData<vec2>(),
			cu_buffers.prev_frame_bilinear_samples_validity_mask.getTypedData<unsigned char>(),
			cu_buffers.albedo_buffer.getTypedData<float>(),
			cu_buffers.noisefree_1spp_acc_tonemapped.getTypedData<float>(),
			cu_buffers.spp_buffer.current().getTypedData<unsigned char>(),
			cu_buffers.noisefree_1spp_accumulated.previous().getTypedData<float>(), 
			cu_buffers.noisefree_1spp_accumulated.current().getTypedData<float>(),
			frame
		);

		// Check results against reference
		//CompareOpenCLBufferAndCudaBuffer("noisefree_1spp_accumulated", cl_buffers.noisefree_1spp_accumulated.current(), cu_buffers.noisefree_1spp_accumulated.current(), command_queue);
		//CompareOpenCLBufferAndCudaBuffer("noisefree_1spp_acc_tonemapped", cl_buffers.noisefree_1spp_acc_tonemapped, cu_buffers.noisefree_1spp_acc_tonemapped, command_queue);

		//CopyOpenCLBufferToCudaBuffer(cl_buffers.noisefree_1spp_acc_tonemapped, cu_buffers.noisefree_1spp_acc_tonemapped, command_queue);
		//CopyOpenCLBufferToCudaBuffer(cl_buffers.noisefree_1spp_accumulated.current(), cu_buffers.noisefree_1spp_accumulated.current(), command_queue);

		#if SAVE_INTERMEDIARY_BUFFERS
		SaveDevice3Float32ImageToDisk("noisefree_1spp_accumulated", frame, command_queue, *cl_buffers.noisefree_1spp_accumulated.current().data(), GetNoiseFree1sppAccumulatedBufferDesc(w, h));
		SaveDevice3Float32ImageToDisk("noisefree_1spp_accumulated", frame, cu_buffers.noisefree_1spp_accumulated.current(), GetNoiseFree1sppAccumulatedBufferDesc(w, h));
		#endif

		// Phase III: Temporal antialiasing
		arg_index = 2;
		K_OPENCL_CHECK(clSetKernelArg(taa_kernel, arg_index++, sizeof(cl_mem), cl_buffers.result_buffer.current().data()));	// [out] Antialiased frame color buffer
		K_OPENCL_CHECK(clSetKernelArg(taa_kernel, arg_index++, sizeof(cl_mem), cl_buffers.result_buffer.previous().data()));	// [in]  Previous frame color buffer
		K_OPENCL_CHECK(clSetKernelArg(taa_kernel, arg_index++, sizeof(cl_int), &frame));	// [in]  Current frame number
		K_OPENCL_CHECK(clEnqueueNDRangeKernel(command_queue, taa_kernel, 2, NULL, k_workset_global_size, k_local_size, 0, NULL, NULL));

		TAAKernelParams params;
		params.sizeX = w;
		params.sizeY = h;
		params.frameNumber = frame;

		run_taa(
			k_workset_grid_size,
			k_block_size,
			params,
			cu_buffers.prev_frame_pixel_coords_buffer.getTypedData<vec2>(),
			cu_buffers.noisefree_1spp_acc_tonemapped.getTypedData<float>(),
			cu_buffers.result_buffer.current().getTypedData<float>(),
			cu_buffers.result_buffer.previous().getTypedData<float>()
		);

		// Check results against reference
		CompareOpenCLBufferAndCudaBuffer("result", cl_buffers.result_buffer.current(), cu_buffers.result_buffer.current(), command_queue);

		#if ENABLE_DEBUG_OUTPUT_TMP_DATA
		if(frame == DEBUG_OUTPUT_FRAME_NUMBER)
		{
			// OpenCL
			{
				const size_t normals_buffer_size = cl_buffers.normals_buffer.current().size();
				openclTmpData.normals.resize(normals_buffer_size / sizeof(float));
				K_OPENCL_CHECK(clEnqueueReadBuffer(command_queue, *cl_buffers.normals_buffer.current().data(), false, 0, normals_buffer_size, openclTmpData.normals.data(), 0, NULL, NULL));

				const size_t positions_buffer_size = cl_buffers.positions_buffer.current().size();
				openclTmpData.positions.resize(positions_buffer_size / sizeof(float));
				K_OPENCL_CHECK(clEnqueueReadBuffer(command_queue, *cl_buffers.positions_buffer.current().data(), false, 0, positions_buffer_size, openclTmpData.positions.data(), 0, NULL, NULL));

				const size_t noisy_1spp_buffer_size = cl_buffers.noisy_1spp_buffer.current().size();
				openclTmpData.noisy_1spp.resize(noisy_1spp_buffer_size / sizeof(float));
				K_OPENCL_CHECK(clEnqueueReadBuffer(command_queue, *cl_buffers.noisy_1spp_buffer.current().data(), false, 0, noisy_1spp_buffer_size, openclTmpData.noisy_1spp.data(), 0, NULL, NULL));

				const size_t prev_frame_pixel_coords_buffer_size = cl_buffers.prev_frame_pixel_coords_buffer.size();
				openclTmpData.prev_frame_pixel_coords_buffer.resize(prev_frame_pixel_coords_buffer_size / sizeof(float));
				K_OPENCL_CHECK(clEnqueueReadBuffer(command_queue, *cl_buffers.prev_frame_pixel_coords_buffer.data(), false, 0, prev_frame_pixel_coords_buffer_size, openclTmpData.prev_frame_pixel_coords_buffer.data(), 0, NULL, NULL));

				const size_t prev_frame_bilinear_samples_validity_mask_size = cl_buffers.prev_frame_bilinear_samples_validity_mask.size();
				openclTmpData.prev_frame_bilinear_samples_validity_mask.resize(prev_frame_bilinear_samples_validity_mask_size / sizeof(unsigned char));
				K_OPENCL_CHECK(clEnqueueReadBuffer(command_queue, *cl_buffers.prev_frame_bilinear_samples_validity_mask.data(), false, 0, prev_frame_bilinear_samples_validity_mask_size, openclTmpData.prev_frame_bilinear_samples_validity_mask.data(), 0, NULL, NULL));

				const size_t spp_buffer_size = cl_buffers.spp_buffer.current().size();
				openclTmpData.spp.resize(spp_buffer_size / sizeof(unsigned char));
				K_OPENCL_CHECK(clEnqueueReadBuffer(command_queue, *cl_buffers.spp_buffer.current().data(), false, 0, spp_buffer_size, openclTmpData.spp.data(), 0, NULL, NULL));

				const size_t features_buffer_size = cl_buffers.features_buffer.size();
				openclTmpData.features_buffer1.resize(features_buffer_size / sizeof(float));
				K_OPENCL_CHECK(clEnqueueReadBuffer(command_queue, *cl_buffers.features_buffer.data(), false, 0, features_buffer_size, openclTmpData.features_buffer1.data(), 0, NULL, NULL));

				const size_t features_weights_buffer_size = cl_buffers.features_weights_buffer.size();
				openclTmpData.features_weights_buffer.resize(features_weights_buffer_size / sizeof(float));
				K_OPENCL_CHECK(clEnqueueReadBuffer(command_queue, *cl_buffers.features_weights_buffer.data(), false, 0, features_weights_buffer_size, openclTmpData.features_weights_buffer.data(), 0, NULL, NULL));

				const size_t features_min_max_buffer_size = cl_buffers.features_min_max_buffer.size();
				openclTmpData.features_min_max_buffer.resize(features_min_max_buffer_size / sizeof(float));
				K_OPENCL_CHECK(clEnqueueReadBuffer(command_queue, *cl_buffers.features_min_max_buffer.data(), false, 0, features_min_max_buffer_size, openclTmpData.features_min_max_buffer.data(), 0, NULL, NULL));

				const size_t noisefree_1spp_size = cl_buffers.noisefree_1spp.size();
				openclTmpData.noisefree_1spp.resize(noisefree_1spp_size / sizeof(float));
				K_OPENCL_CHECK(clEnqueueReadBuffer(command_queue, *cl_buffers.noisefree_1spp.data(), false, 0, noisefree_1spp_size, openclTmpData.noisefree_1spp.data(), 0, NULL, NULL));

				const size_t noisefree_1spp_accumulated_size = cl_buffers.noisefree_1spp_accumulated.current().size();
				openclTmpData.noisefree_1spp_accumulated.resize(noisefree_1spp_accumulated_size / sizeof(float));
				K_OPENCL_CHECK(clEnqueueReadBuffer(command_queue, *cl_buffers.noisefree_1spp_accumulated.current().data(), false, 0, noisefree_1spp_accumulated_size, openclTmpData.noisefree_1spp_accumulated.data(), 0, NULL, NULL));

				const size_t noisefree_1spp_acc_tonemapped_size = cl_buffers.noisefree_1spp_acc_tonemapped.size();
				openclTmpData.noisefree_1spp_acc_tonemapped.resize(noisefree_1spp_acc_tonemapped_size / sizeof(float));
				K_OPENCL_CHECK(clEnqueueReadBuffer(command_queue, *cl_buffers.noisefree_1spp_acc_tonemapped.data(), false, 0, noisefree_1spp_acc_tonemapped_size, openclTmpData.noisefree_1spp_acc_tonemapped.data(), 0, NULL, NULL));

				const size_t result_buffer_size = cl_buffers.result_buffer.current().size();
				openclTmpData.result.resize(result_buffer_size / sizeof(float));
				K_OPENCL_CHECK(clEnqueueReadBuffer(command_queue, *cl_buffers.result_buffer.current().data(), false, 0, result_buffer_size, openclTmpData.result.data(), 0, NULL, NULL));

				K_OPENCL_CHECK(clFlush(command_queue));
				K_OPENCL_CHECK(clFinish(command_queue));
			}

			// Cuda
			{
				const size_t normals_buffer_size = cu_buffers.normals_buffer.current().size();
				cudaTmpData.normals.resize(normals_buffer_size / sizeof(float));
				K_CUDA_CHECK(cudaMemcpy(cudaTmpData.normals.data(), cu_buffers.normals_buffer.current().data(), normals_buffer_size, cudaMemcpyDeviceToHost));

				const size_t positions_buffer_size = cu_buffers.positions_buffer.current().size();
				cudaTmpData.positions.resize(positions_buffer_size / sizeof(float));
				K_CUDA_CHECK(cudaMemcpy(cudaTmpData.positions.data(), cu_buffers.positions_buffer.current().data(), positions_buffer_size, cudaMemcpyDeviceToHost));

				const size_t noisy_1spp_buffer_size = cu_buffers.noisy_1spp_buffer.current().size();
				cudaTmpData.noisy_1spp.resize(noisy_1spp_buffer_size / sizeof(float));
				K_CUDA_CHECK(cudaMemcpy(cudaTmpData.noisy_1spp.data(), cu_buffers.noisy_1spp_buffer.current().data(), noisy_1spp_buffer_size, cudaMemcpyDeviceToHost));

				const size_t prev_frame_pixel_coords_buffer_size = cu_buffers.prev_frame_pixel_coords_buffer.size();
				cudaTmpData.prev_frame_pixel_coords_buffer.resize(prev_frame_pixel_coords_buffer_size / sizeof(float));
				K_CUDA_CHECK(cudaMemcpy(cudaTmpData.prev_frame_pixel_coords_buffer.data(), cu_buffers.prev_frame_pixel_coords_buffer.data(), prev_frame_pixel_coords_buffer_size, cudaMemcpyDeviceToHost));

				const size_t prev_frame_bilinear_samples_validity_mask_size = cu_buffers.prev_frame_bilinear_samples_validity_mask.size();
				cudaTmpData.prev_frame_bilinear_samples_validity_mask.resize(prev_frame_bilinear_samples_validity_mask_size / sizeof(unsigned char));
				K_CUDA_CHECK(cudaMemcpy(cudaTmpData.prev_frame_bilinear_samples_validity_mask.data(), cu_buffers.prev_frame_bilinear_samples_validity_mask.data(), prev_frame_bilinear_samples_validity_mask_size, cudaMemcpyDeviceToHost));

				const size_t spp_buffer_size = cu_buffers.spp_buffer.current().size();
				cudaTmpData.spp.resize(spp_buffer_size / sizeof(unsigned char));
				K_CUDA_CHECK(cudaMemcpy(cudaTmpData.spp.data(), cu_buffers.spp_buffer.current().data(), spp_buffer_size, cudaMemcpyDeviceToHost));

				const size_t features_buffer_size = cu_buffers.features_buffer.size();
				cudaTmpData.features_buffer1.resize(features_buffer_size / sizeof(float));
				K_CUDA_CHECK(cudaMemcpy(cudaTmpData.features_buffer1.data(), cu_buffers.features_buffer.data(), features_buffer_size, cudaMemcpyDeviceToHost));

				const size_t features_weights_buffer_size = cu_buffers.features_weights_buffer.size();
				cudaTmpData.features_weights_buffer.resize(features_weights_buffer_size / sizeof(float));
				K_CUDA_CHECK(cudaMemcpy(cudaTmpData.features_weights_buffer.data(), cu_buffers.features_weights_buffer.data(), features_weights_buffer_size, cudaMemcpyDeviceToHost));

				const size_t features_min_max_buffer_size = cu_buffers.features_min_max_buffer.size();
				cudaTmpData.features_min_max_buffer.resize(features_min_max_buffer_size / sizeof(float));
				K_CUDA_CHECK(cudaMemcpy(cudaTmpData.features_min_max_buffer.data(), cu_buffers.features_min_max_buffer.data(), features_min_max_buffer_size, cudaMemcpyDeviceToHost));

				const size_t noisefree_1spp_size = cu_buffers.noisefree_1spp.size();
				cudaTmpData.noisefree_1spp.resize(noisefree_1spp_size / sizeof(float));
				K_CUDA_CHECK(cudaMemcpy(cudaTmpData.noisefree_1spp.data(), cu_buffers.noisefree_1spp.data(), noisefree_1spp_size, cudaMemcpyDeviceToHost));

				const size_t noisefree_1spp_accumulated_size = cu_buffers.noisefree_1spp_accumulated.current().size();
				cudaTmpData.noisefree_1spp_accumulated.resize(noisefree_1spp_accumulated_size / sizeof(float));
				K_CUDA_CHECK(cudaMemcpy(cudaTmpData.noisefree_1spp_accumulated.data(), cu_buffers.noisefree_1spp_accumulated.current().data(), noisefree_1spp_accumulated_size, cudaMemcpyDeviceToHost));

				const size_t noisefree_1spp_acc_tonemapped_size = cu_buffers.noisefree_1spp_acc_tonemapped.size();
				cudaTmpData.noisefree_1spp_acc_tonemapped.resize(noisefree_1spp_acc_tonemapped_size / sizeof(float));
				K_CUDA_CHECK(cudaMemcpy(cudaTmpData.noisefree_1spp_acc_tonemapped.data(), cu_buffers.noisefree_1spp_acc_tonemapped.data(), noisefree_1spp_acc_tonemapped_size, cudaMemcpyDeviceToHost));

				const size_t result_buffer_size = cu_buffers.result_buffer.current().size();
				cudaTmpData.result.resize(result_buffer_size / sizeof(float));
				K_CUDA_CHECK(cudaMemcpy(cudaTmpData.result.data(), cu_buffers.result_buffer.current().data(), result_buffer_size, cudaMemcpyDeviceToHost));

				K_CUDA_CHECK(cudaDeviceSynchronize());
			}
			
			return;
		}
		#endif

		#if SAVE_FINAL_RESULT
		SaveDevice3Float32ImageToDisk("result", frame, command_queue, *cl_buffers.result_buffer.current().data(), GetResultBufferDesc(w, h));
		SaveDevice3Float32ImageToDisk("result", frame, cu_buffers.result_buffer.current(), GetResultBufferDesc(w, h));
		#endif

		//CopyOpenCLBufferToCudaBuffer(cl_buffers.result_buffer.current(), cu_buffers.result_buffer.current(), command_queue);

		// Swap all double buffers
		std::for_each(opencl_double_buffers.begin(), opencl_double_buffers.end(), std::bind(&Double_buffer<OpenCLDeviceBuffer>::swap, std::placeholders::_1));
		std::for_each(cuda_double_buffers.begin(), cuda_double_buffers.end(), std::bind(&Double_buffer<CudaDeviceBuffer>::swap, std::placeholders::_1));
	}
    
	// Clean up ////////////////////////////////////////////////////////////////

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
}