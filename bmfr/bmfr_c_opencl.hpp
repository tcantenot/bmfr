#pragma once

#include "config.hpp"
#include "opencl_helpers.hpp"

struct BMFROpenCLBuffers
{
	OpenCLDeviceBuffer albedo_buffer;
	Double_buffer<OpenCLDeviceBuffer> normals_buffer;
	Double_buffer<OpenCLDeviceBuffer> positions_buffer;
	Double_buffer<OpenCLDeviceBuffer> noisy_1spp_buffer;
	OpenCLDeviceBuffer prev_frame_pixel_coords_buffer;
	OpenCLDeviceBuffer prev_frame_bilinear_samples_validity_mask;
	Double_buffer<OpenCLDeviceBuffer> spp_buffer;
	OpenCLDeviceBuffer features_buffer;
	OpenCLDeviceBuffer features_weights_buffer;
	OpenCLDeviceBuffer features_min_max_buffer;
	OpenCLDeviceBuffer noisefree_1spp;
	Double_buffer<OpenCLDeviceBuffer> noisefree_1spp_accumulated;
	OpenCLDeviceBuffer noisefree_1spp_acc_tonemapped;
	Double_buffer<OpenCLDeviceBuffer> result_buffer;

	std::vector<float> result_host;
};

void init_opencl(cl_context & context, cl_command_queue & command_queue, cl_device_id & device);
void init_bmfr_opencl_buffers(BMFROpenCLBuffers & buffers, size_t w, size_t h, size_t features_count, cl_context c);


int bmfr_c_opencl(TmpData & tmpData);
