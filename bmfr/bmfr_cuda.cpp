#include "bmfr_cuda.hpp"
#include "bmfr.cuh"

#include "cuda_helpers.hpp"
#include "utils.hpp"
#include <functional>


void init_bmfr_cuda_buffers(BMFRCudaBuffers & buffers, size_t w, size_t h, size_t features_count)
{
	size_t cudaBufferTotalSize = 0;

	// Albedo buffer (3 * float32) // TODO: compress this
	const BufferDesc albedoBufferDesc = GetAlbedoBufferDesc(w, h);
	const size_t albedo_buffer_size = albedoBufferDesc.byte_size;
    buffers.albedo_buffer.init(albedo_buffer_size);
	cudaBufferTotalSize += albedo_buffer_size;

	// (World) normals buffers (3 * float32) in [-1, +1]^3
	const BufferDesc normalsBufferDesc = GetNormalsBufferDesc(w, h);
	const size_t normals_buffer_size = normalsBufferDesc.byte_size;
	buffers.normals_buffer[0].init(normals_buffer_size);
	buffers.normals_buffer[1].init(normals_buffer_size);
	cudaBufferTotalSize += 2 * normals_buffer_size;

	// World positions buffer (3 * float32)
	// TODO: normalize in [0, 1] (or [-1, +1])
	const BufferDesc positionsBufferDesc = GetPositionsBufferDesc(w, h);
	const size_t positions_buffer_size = positionsBufferDesc.byte_size;
    buffers.positions_buffer[0].init(positions_buffer_size);
    buffers.positions_buffer[1].init(positions_buffer_size);
	cudaBufferTotalSize += 2 * positions_buffer_size;

	// Frame noisy 1spp color buffer (3 * float32)
	const BufferDesc noisy1sppBufferDesc = GetNoisy1sppBufferDesc(w, h);
	const size_t noisy_1spp_buffer_size = noisy1sppBufferDesc.byte_size;
	buffers.frame_noisy_1spp_buffer.init(noisy_1spp_buffer_size);
	cudaBufferTotalSize += noisy_1spp_buffer_size;
    
	// Accumulated noisy 1spp  color buffer (3 * float32)
	buffers.noisy_1spp_buffer[0].init(noisy_1spp_buffer_size);
	buffers.noisy_1spp_buffer[1].init(noisy_1spp_buffer_size);
	cudaBufferTotalSize += 2 * noisy_1spp_buffer_size;

	// Features buffer (half or single-precision) (3 * float16 or 3 * float32)
	const BufferDesc featuresBufferDesc = GetFeaturesBufferDesc(w, h, features_count - 3, USE_HALF_PRECISION_IN_FEATURES_DATA);
	const size_t features_buffer_size = featuresBufferDesc.byte_size;
    buffers.features_buffer.init(features_buffer_size);
	cudaBufferTotalSize += features_buffer_size;

	// Noise-free color estimate (3 * float32)
	const BufferDesc noiseFree1sppBufferDesc = GetNoiseFree1sppBufferDesc(w, h);
	const size_t noisefree_1spp_size = noiseFree1sppBufferDesc.byte_size;
    buffers.noisefree_1spp.init(noisefree_1spp_size);
	cudaBufferTotalSize += noisefree_1spp_size;

	// Noise-free accumulated color estimate (3 * float32)
	const BufferDesc noiseFree1sppAccumulatedBufferDesc = GetNoiseFree1sppAccumulatedBufferDesc(w, h);
	const size_t noisefree_1spp_accumulated_size = noiseFree1sppAccumulatedBufferDesc.byte_size;
    buffers.noisefree_1spp_accumulated[0].init(noisefree_1spp_accumulated_size);
    buffers.noisefree_1spp_accumulated[1].init(noisefree_1spp_accumulated_size);
	cudaBufferTotalSize += 2 * noisefree_1spp_accumulated_size;

	// Final antialiased color buffer (3 * float32)
	const BufferDesc resultBufferDesc = GetResultBufferDesc(w, h);
	const size_t result_buffer_size = resultBufferDesc.byte_size;
    buffers.result_buffer[0].init(result_buffer_size);
    buffers.result_buffer[1].init(result_buffer_size);
	cudaBufferTotalSize += 2 * result_buffer_size;

	// Previous frame pixel coordinates (after reprojection) (2 * float32)
	const BufferDesc prevFramePixelCoordsBufferDesc = GetPrevFramePixelCoordsBufferDesc(w, h);
	const size_t prev_frame_pixel_coords_buffer_size = prevFramePixelCoordsBufferDesc.byte_size;
    buffers.prev_frame_pixel_coords_buffer.init(prev_frame_pixel_coords_buffer_size);
	cudaBufferTotalSize += prev_frame_pixel_coords_buffer_size;

	// Validity mask of reprojected bilinear samples into previous frame (uchar 8bits)
	const BufferDesc prevFrameBilinearSamplesValidityMaskBufferDesc = GetPrevFrameBilinearSamplesValidityMaskBufferDesc(w, h);
	const size_t prev_frame_bilinear_samples_validity_mask_size = prevFrameBilinearSamplesValidityMaskBufferDesc.byte_size;
    buffers.prev_frame_bilinear_samples_validity_mask.init(prev_frame_bilinear_samples_validity_mask_size);
	cudaBufferTotalSize += prev_frame_bilinear_samples_validity_mask_size;

	// Tonemapped noise-free color estimate w/ albedo (3 * float32)
	const BufferDesc noiseFree1sppAccTonemappedBufferDesc = GetNoiseFree1sppAccTonemappedBufferDesc(w, h);
	const size_t noisefree_1spp_acc_tonemapped_size = noiseFree1sppAccTonemappedBufferDesc.byte_size;
    buffers.noisefree_1spp_acc_tonemapped.init(noisefree_1spp_acc_tonemapped_size);
	cudaBufferTotalSize += noisefree_1spp_acc_tonemapped_size;

	// Features weights per color channel (x3) (computed by the BMFR) (3 * float32)
	const BufferDesc featuresWeightsBufferDesc = GetFeaturesWeightsBufferDesc(w, h, (features_count - 3));
	const size_t features_weights_buffer_size = featuresWeightsBufferDesc.byte_size;
    buffers.features_weights_buffer.init(features_weights_buffer_size);
	cudaBufferTotalSize += features_weights_buffer_size;

	// Min and max of features values per block (world_positions) (6 * 2 * float32)
	const BufferDesc featuresMinMaxBufferDesc = GetFeaturesMinMaxBufferDesc(w, h, FEATURES_SCALED);
	const size_t features_min_max_buffer_size = featuresMinMaxBufferDesc.byte_size;
    buffers.features_min_max_buffer.init(features_min_max_buffer_size);
	cudaBufferTotalSize += features_min_max_buffer_size;

	// Number of samples accumulated (for cumulative moving average) (char 8bits)
	const BufferDesc sppBufferDesc = GetSppBufferDesc(w, h);
	const size_t spp_buffer_size = sppBufferDesc.byte_size;
    buffers.spp_buffer[0].init(spp_buffer_size);
    buffers.spp_buffer[1].init(spp_buffer_size);
	cudaBufferTotalSize += 2 * spp_buffer_size;

    buffers.result_host.resize(3 * w * h);

	LOG("CUDA buffers total size: %.3fMB\n", float(cudaBufferTotalSize) / 1024.f / 1024.f);
}

int bmfr_cuda(TmpData & tmpData)
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
    const auto r_size = ((buffer_count - 2) * (buffer_count - 1) / 2) * sizeof(vec3);
	#else
    const auto r_size = (buffer_count - 2) * (buffer_count - 2) * sizeof(vec3);
	#endif

	const size_t w = IMAGE_WIDTH;
	const size_t h = IMAGE_HEIGHT;
	
	const size_t localWidth					= GetLocalWidth();
	const size_t localHeight				= GetLocalHeight();
	const size_t worksetWidth				= ComputeWorksetWidth(w);
	const size_t worksetHeight				= ComputeWorksetHeight(h);
	const size_t worksetWidthWithMargin		= ComputeWorksetWithMarginWidth(w);
	const size_t worksetHeightWithMargin	= ComputeWorksetWithMarginHeight(h);
	const size_t fitterLocalSize			= GetFitterLocalSize();
	const size_t fitterGlobalSize			= GetFitterGlobalSize();


	// Create CUDA buffers

	LOG("\nAllocate CUDA buffers\n");

	BMFRCudaBuffers buffers;
	init_bmfr_cuda_buffers(buffers, w, h, buffer_count);

	std::vector<Double_buffer<CudaDeviceBuffer> *> cuda_double_buffers =
	{
		&buffers.normals_buffer,
		&buffers.positions_buffer,
		&buffers.noisy_1spp_buffer,
		&buffers.noisefree_1spp_accumulated,
		&buffers.result_buffer,
		&buffers.spp_buffer
	};

	std::vector<CudaTimer> frame_timers(FRAME_COUNT);
	std::vector<CudaTimer> accumulate_noisy_data_timers(FRAME_COUNT);
	std::vector<CudaTimer> fitter_timers(FRAME_COUNT);
	std::vector<CudaTimer> weighted_sum_timers(FRAME_COUNT);
	std::vector<CudaTimer> accumulate_noisefree_estimate_timers(FRAME_COUNT);
	std::vector<CudaTimer> taa_timers(FRAME_COUNT);

    const dim3 k_block_size(localWidth, localHeight);
    const dim3 k_workset_grid_size((worksetWidth + k_block_size.x - 1) / k_block_size.x, (worksetHeight + k_block_size.y - 1) / k_block_size.y);
	const dim3 k_workset_with_margin_grid_size((worksetWidthWithMargin + k_block_size.x - 1) / k_block_size.x, (worksetHeightWithMargin + k_block_size.y - 1) / k_block_size.y);
    const dim3 k_fitter_block_size(fitterLocalSize);
    const dim3 k_fitter_grid_size((fitterGlobalSize + k_fitter_block_size.x - 1) / k_fitter_block_size.x);
	
    LOG("\nRun kernels.\n");

	FrameInputData frameInput;
	for(int frame = 0; frame < FRAME_COUNT; ++frame)
    {
		LOG("Frame %d\n", frame);

		LOG("  Load frame input data from disk\n");
		LoadFrameInputData(frameInput, w, h, frame);

		LOG("  Transfert data from host to device\n");
		K_CUDA_CHECK(cudaMemcpy(buffers.albedo_buffer.data(), frameInput.albedos.data(), frameInput.albedos.size() * sizeof(float), cudaMemcpyHostToDevice));
        K_CUDA_CHECK(cudaMemcpy(buffers.normals_buffer.current().data(), frameInput.normals.data(), frameInput.normals.size() * sizeof(float), cudaMemcpyHostToDevice));
        K_CUDA_CHECK(cudaMemcpy(buffers.positions_buffer.current().data(), frameInput.positions.data(), frameInput.positions.size() * sizeof(float), cudaMemcpyHostToDevice));
        K_CUDA_CHECK(cudaMemcpy(buffers.frame_noisy_1spp_buffer.data(), frameInput.noisy1spps.data(), frameInput.noisy1spps.size() * sizeof(float), cudaMemcpyHostToDevice));

		frame_timers[frame].start();

		// Phase I:
		//  - accumulate noisy 1spp
		//  - compute previous frame pixel coordinates (after reprojection)
		//  - generate validity bit mask of bilinear samples of previous frame
		//  - concatenate the different features in a single buffer
        // Note: On the first frame accum_noisy_kernel just copies to the features_buffer
        
		DEBUG_LOG("  Run accumulate_noisy_data kernel: grid (%d, %d) | block(%d, %d)\n", 
			k_workset_with_margin_grid_size.x,
			k_workset_with_margin_grid_size.y,
			k_block_size.x,
			k_block_size.y
		);

		accumulate_noisy_data_timers[frame].start();
        const int matrix_index = frame == 0 ? 0 : frame - 1;
		const mat4x4 cam_mat = *reinterpret_cast<mat4x4 const *>(&camera_matrices[matrix_index][0][0]);
		const vec2 pix_off = *reinterpret_cast<vec2 const *>(&pixel_offsets[frame][0]);
		run_accumulate_noisy_data(
			k_workset_with_margin_grid_size,
			k_block_size,
			buffers.prev_frame_pixel_coords_buffer.getTypedData<vec2>(),
			buffers.prev_frame_bilinear_samples_validity_mask.getTypedData<unsigned char>(),
			buffers.normals_buffer.current().getTypedData<float>(),
			buffers.normals_buffer.previous().getTypedData<float>(),
			buffers.positions_buffer.current().getTypedData<float>(),
			buffers.positions_buffer.previous().getTypedData<float>(),
			buffers.frame_noisy_1spp_buffer.getTypedData<float>(),
			buffers.noisy_1spp_buffer.current().getTypedData<float>(),
			buffers.noisy_1spp_buffer.previous().getTypedData<float>(),
			// TODO: invert the order of the spp buffers
			buffers.spp_buffer.previous().getTypedData<unsigned char>(),
			buffers.spp_buffer.current().getTypedData<unsigned char>(),
			buffers.features_buffer.getTypedData<float>(),
			cam_mat,
			pix_off,
			frame
		);
		accumulate_noisy_data_timers[frame].stop();

		#if ENABLE_DEBUG_OUTPUT_TMP_DATA
		if(frame == DEBUG_OUTPUT_FRAME_NUMBER)
		{
			tmpData.features_buffer0.resize(buffers.features_buffer.size() / sizeof(float));
			K_CUDA_CHECK(cudaMemcpy(tmpData.features_buffer0.data(), buffers.features_buffer.data(), buffers.features_buffer.size(), cudaMemcpyDeviceToHost));
		}
		#endif

		//K_CUDA_CHECK(cudaDeviceSynchronize());

		#if SAVE_INTERMEDIARY_BUFFERS
		SaveDevice3Float32ImageToDisk("noisy_1spp_buffer", frame, buffers.noisy_1spp_buffer.current(), GetNoisy1sppBufferDesc(w, h));
		#endif

		// Phase II: Blockwise Multi-Order Feature Regression (BMFR)
		// -> compute features weights

		DEBUG_LOG("  Run fitter kernel: grid (%d, %d) | block(%d, %d)\n", 
			k_fitter_grid_size.x,
			k_fitter_grid_size.y,
			k_fitter_block_size.x,
			k_fitter_block_size.y
		);

		fitter_timers[frame].start();
		run_fitter(
			k_fitter_grid_size,
			k_fitter_block_size,
			buffers.features_weights_buffer.getTypedData<float>(),
			buffers.features_min_max_buffer.getTypedData<float>(),
			buffers.features_buffer.getTypedData<float>(),
			frame
		);
		fitter_timers[frame].stop();

		//K_CUDA_CHECK(cudaDeviceSynchronize());

		// Phase II: Compute noise free color estimate (weighted sum of features)

		DEBUG_LOG("  Run weighted_sum kernel: grid (%d, %d) | block(%d, %d)\n", 
			k_workset_grid_size.x,
			k_workset_grid_size.y,
			k_block_size.x,
			k_block_size.y
		);

		weighted_sum_timers[frame].start();
		run_weighted_sum(
			k_workset_grid_size,
			k_block_size,
			buffers.features_weights_buffer.getTypedData<float>(),
			buffers.features_min_max_buffer.getTypedData<float>(),
			buffers.noisefree_1spp.getTypedData<float>(),
			buffers.normals_buffer.current().getTypedData<float>(),
			buffers.positions_buffer.current().getTypedData<float>(),
			buffers.noisy_1spp_buffer.current().getTypedData<float>(),
			frame
		);
		weighted_sum_timers[frame].stop();

		//K_CUDA_CHECK(cudaDeviceSynchronize());

		#if SAVE_INTERMEDIARY_BUFFERS
		SaveDevice3Float32ImageToDisk("noisefree_1spp", frame, buffers.noisefree_1spp, GetNoiseFree1sppBufferDesc(w, h));
		#endif

		// Phase III: Postprocessing
		// -> accumulate noise-free color estimate + output a tonemapped version

		DEBUG_LOG("  Run accumulate_filtered_data kernel: grid (%d, %d) | block(%d, %d)\n", 
			k_workset_grid_size.x,
			k_workset_grid_size.y,
			k_block_size.x,
			k_block_size.y
		);

		accumulate_noisefree_estimate_timers[frame].start();
		run_accumulate_filtered_data(
			k_workset_grid_size,
			k_block_size,
			buffers.noisefree_1spp.getTypedData<float>(),
			buffers.prev_frame_pixel_coords_buffer.getTypedData<vec2>(),
			buffers.prev_frame_bilinear_samples_validity_mask.getTypedData<unsigned char>(),
			buffers.albedo_buffer.getTypedData<float>(),
			buffers.noisefree_1spp_acc_tonemapped.getTypedData<float>(),
			buffers.spp_buffer.current().getTypedData<unsigned char>(),
			buffers.noisefree_1spp_accumulated.previous().getTypedData<float>(), 
			buffers.noisefree_1spp_accumulated.current().getTypedData<float>(),
			frame
		);
		accumulate_noisefree_estimate_timers[frame].stop();

		#if SAVE_INTERMEDIARY_BUFFERS
		SaveDevice3Float32ImageToDisk("noisefree_1spp_accumulated", frame, buffers.noisefree_1spp_accumulated.current(), GetNoiseFree1sppAccumulatedBufferDesc(w, h));
		#endif

		// Phase III: Temporal antialiasing

		DEBUG_LOG("  Run taa kernel: grid (%d, %d) | block(%d, %d)\n", 
			k_workset_grid_size.x,
			k_workset_grid_size.y,
			k_block_size.x,
			k_block_size.y
		);

		taa_timers[frame].start();
		run_taa(
			k_workset_grid_size,
			k_block_size,
			buffers.prev_frame_pixel_coords_buffer.getTypedData<vec2>(),
			buffers.noisefree_1spp_acc_tonemapped.getTypedData<float>(),
			buffers.result_buffer.current().getTypedData<float>(),
			buffers.result_buffer.previous().getTypedData<float>(),
			frame
		);
		taa_timers[frame].stop();
		frame_timers[frame].stop();

		#if ENABLE_DEBUG_OUTPUT_TMP_DATA
		if(frame == DEBUG_OUTPUT_FRAME_NUMBER)
		{
			const size_t normals_buffer_size = buffers.normals_buffer.current().size();
			tmpData.normals.resize(normals_buffer_size / sizeof(float));
			K_CUDA_CHECK(cudaMemcpy(tmpData.normals.data(), buffers.normals_buffer.current().data(), normals_buffer_size, cudaMemcpyDeviceToHost));

			const size_t positions_buffer_size = buffers.positions_buffer.current().size();
			tmpData.positions.resize(positions_buffer_size / sizeof(float));
			K_CUDA_CHECK(cudaMemcpy(tmpData.positions.data(), buffers.positions_buffer.current().data(), positions_buffer_size, cudaMemcpyDeviceToHost));

			const size_t noisy_1spp_buffer_size = buffers.noisy_1spp_buffer.current().size();
			tmpData.noisy_1spp.resize(noisy_1spp_buffer_size / sizeof(float));
			K_CUDA_CHECK(cudaMemcpy(tmpData.noisy_1spp.data(), buffers.noisy_1spp_buffer.current().data(), noisy_1spp_buffer_size, cudaMemcpyDeviceToHost));

			const size_t prev_frame_pixel_coords_buffer_size = buffers.prev_frame_pixel_coords_buffer.size();
			tmpData.prev_frame_pixel_coords_buffer.resize(prev_frame_pixel_coords_buffer_size / sizeof(float));
			K_CUDA_CHECK(cudaMemcpy(tmpData.prev_frame_pixel_coords_buffer.data(), buffers.prev_frame_pixel_coords_buffer.data(), prev_frame_pixel_coords_buffer_size, cudaMemcpyDeviceToHost));

			const size_t prev_frame_bilinear_samples_validity_mask_size = buffers.prev_frame_bilinear_samples_validity_mask.size();
			tmpData.prev_frame_bilinear_samples_validity_mask.resize(prev_frame_bilinear_samples_validity_mask_size / sizeof(unsigned char));
			K_CUDA_CHECK(cudaMemcpy(tmpData.prev_frame_bilinear_samples_validity_mask.data(), buffers.prev_frame_bilinear_samples_validity_mask.data(), prev_frame_bilinear_samples_validity_mask_size, cudaMemcpyDeviceToHost));

			const size_t spp_buffer_size = buffers.spp_buffer.current().size();
			tmpData.spp.resize(spp_buffer_size / sizeof(unsigned char));
			K_CUDA_CHECK(cudaMemcpy(tmpData.spp.data(), buffers.spp_buffer.current().data(), spp_buffer_size, cudaMemcpyDeviceToHost));

			const size_t features_buffer_size = buffers.features_buffer.size();
			tmpData.features_buffer1.resize(features_buffer_size / sizeof(float));
			K_CUDA_CHECK(cudaMemcpy(tmpData.features_buffer1.data(), buffers.features_buffer.data(), features_buffer_size, cudaMemcpyDeviceToHost));

			const size_t features_weights_buffer_size = buffers.features_weights_buffer.size();
			tmpData.features_weights_buffer.resize(features_weights_buffer_size / sizeof(float));
			K_CUDA_CHECK(cudaMemcpy(tmpData.features_weights_buffer.data(), buffers.features_weights_buffer.data(), features_weights_buffer_size, cudaMemcpyDeviceToHost));

			const size_t features_min_max_buffer_size = buffers.features_min_max_buffer.size();
			tmpData.features_min_max_buffer.resize(features_min_max_buffer_size / sizeof(float));
			K_CUDA_CHECK(cudaMemcpy(tmpData.features_min_max_buffer.data(), buffers.features_min_max_buffer.data(), features_min_max_buffer_size, cudaMemcpyDeviceToHost));

			const size_t noisefree_1spp_size = buffers.noisefree_1spp.size();
			tmpData.noisefree_1spp.resize(noisefree_1spp_size / sizeof(float));
			K_CUDA_CHECK(cudaMemcpy(tmpData.noisefree_1spp.data(), buffers.noisefree_1spp.data(), noisefree_1spp_size, cudaMemcpyDeviceToHost));

			const size_t noisefree_1spp_accumulated_size = buffers.noisefree_1spp_accumulated.current().size();
			tmpData.noisefree_1spp_accumulated.resize(noisefree_1spp_accumulated_size / sizeof(float));
			K_CUDA_CHECK(cudaMemcpy(tmpData.noisefree_1spp_accumulated.data(), buffers.noisefree_1spp_accumulated.current().data(), noisefree_1spp_accumulated_size, cudaMemcpyDeviceToHost));

			const size_t noisefree_1spp_acc_tonemapped_size = buffers.noisefree_1spp_acc_tonemapped.size();
			tmpData.noisefree_1spp_acc_tonemapped.resize(noisefree_1spp_acc_tonemapped_size / sizeof(float));
			K_CUDA_CHECK(cudaMemcpy(tmpData.noisefree_1spp_acc_tonemapped.data(), buffers.noisefree_1spp_acc_tonemapped.data(), noisefree_1spp_acc_tonemapped_size, cudaMemcpyDeviceToHost));

			const size_t result_buffer_size = buffers.result_buffer.current().size();
			tmpData.result.resize(result_buffer_size / sizeof(float));
			K_CUDA_CHECK(cudaMemcpy(tmpData.result.data(), buffers.result_buffer.current().data(), result_buffer_size, cudaMemcpyDeviceToHost));

			K_CUDA_CHECK(cudaDeviceSynchronize());
			
			return 0;
		}
		#endif

		#if SAVE_FINAL_RESULT
		SaveDevice3Float32ImageToDisk("result", frame, buffers.result_buffer.current(), GetResultBufferDesc(w, h));
		#endif

		// Swap all double buffers
        std::for_each(cuda_double_buffers.begin(), cuda_double_buffers.end(), std::bind(&Double_buffer<CudaDeviceBuffer>::swap, std::placeholders::_1));
	}

	float totalElapsedTime_ms = 0.f;
	for(int frame = 0; frame < FRAME_COUNT; ++frame)
	{
		float elapsedTime_ms = frame_timers[frame].elaspedTime();
		LOG("Duration of frame %d: %.3fms\n", frame, elapsedTime_ms);
		LOG("  accumulate noisy data: %.3fms\n", accumulate_noisy_data_timers[frame].elaspedTime());
		LOG("  fitter: %.3fms\n", fitter_timers[frame].elaspedTime());
		LOG("  weighted sum: %.3fms\n", weighted_sum_timers[frame].elaspedTime());
		LOG("  accumulate noise-free estimage: %.3fms\n", accumulate_noisefree_estimate_timers[frame].elaspedTime());
		LOG("  taa: %.3fms\n", taa_timers[frame].elaspedTime());
		totalElapsedTime_ms += elapsedTime_ms;
	}

	float avgFrameTime_ms = totalElapsedTime_ms / float(FRAME_COUNT);
	LOG("Average frame timing: %.3fms\n", avgFrameTime_ms);

	return 0;
}
