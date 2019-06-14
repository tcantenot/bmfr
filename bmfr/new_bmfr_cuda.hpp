#pragma once

#include "config.hpp"
#include "cuda_helpers.hpp"

struct NewBMFRCudaBuffers
{
	CudaDeviceBuffer albedo_buffer;
	Double_buffer<CudaDeviceBuffer> normals_buffer;
	Double_buffer<CudaDeviceBuffer> positions_buffer;
	CudaDeviceBuffer frame_noisy_1spp_buffer;
	Double_buffer<CudaDeviceBuffer> noisy_1spp_buffer;
	CudaDeviceBuffer prev_frame_pixel_coords_buffer;
	CudaDeviceBuffer prev_frame_bilinear_samples_validity_mask;
	Double_buffer<CudaDeviceBuffer> spp_buffer;
	CudaDeviceBuffer features_buffer;
	CudaDeviceBuffer features_weights_buffer;
	CudaDeviceBuffer noisefree_1spp;
	Double_buffer<CudaDeviceBuffer> noisefree_1spp_accumulated;
	CudaDeviceBuffer noisefree_1spp_acc_tonemapped;
	Double_buffer<CudaDeviceBuffer> result_buffer;

	std::vector<float> result_host;
};

void init_new_bmfr_cuda_buffers(NewBMFRCudaBuffers & buffers, size_t w, size_t h, size_t features_count);

int new_bmfr_cuda(TmpData & data);
