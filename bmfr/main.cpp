#include "bmfr.hpp"
#include "bmfr_c_opencl.hpp"
#include "bmfr_cuda.hpp"
#include "bmfr_cuda_opencl.hpp"

#include <assert.h>

// CUDA 8.0 on Visual Studio 2017
//  - Install CUDA 8.0
//  - Copy content of C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\extras\visual_studio_integration\MSBuildExtensions
//	  into C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\Common7\IDE\VC\VCTargets\BuildCustomizations
//  - Select Cuda 8.0 on the project right-click menu "Build Dependencies/Build Customizations..."
int main()
{
	TmpData cuda_tmp_data, opencl_tmp_data;
	//bmfr_c_opencl(opencl_tmp_data);
	//bmfr_cuda(cuda_tmp_data);
	bmfr_cuda_c_opencl(cuda_tmp_data, opencl_tmp_data);

#if !ENABLE_DEBUG_OUTPUT_TMP_DATA
	return 0;
#endif

    //bmfr_opencl();
	//bmfr_c_opencl(opencl_tmp_data);

	printf("\n");

	CheckDiffFloat("normals", cuda_tmp_data.normals, opencl_tmp_data.normals);
	CheckDiffFloat("positions", cuda_tmp_data.positions, opencl_tmp_data.positions);
	//CheckDiffFloat("noisy_1spp", cuda_tmp_data.noisy_1spp, opencl_tmp_data.noisy_1spp);
	CheckDiffFloat("prev_frame_pixel_coords_buffer", cuda_tmp_data.prev_frame_pixel_coords_buffer, opencl_tmp_data.prev_frame_pixel_coords_buffer, false);
	//CheckDiffFloat("features_buffer0", cuda_tmp_data.features_buffer0, opencl_tmp_data.features_buffer0);
	
#if 0
	assert(cuda_tmp_data.prev_frame_bilinear_samples_validity_mask.size() == opencl_tmp_data.prev_frame_bilinear_samples_validity_mask.size());
	for(int i = 0; i < cuda_tmp_data.prev_frame_bilinear_samples_validity_mask.size(); ++i)
	{
		if(cuda_tmp_data.prev_frame_bilinear_samples_validity_mask[i] != opencl_tmp_data.prev_frame_bilinear_samples_validity_mask[i])
		{
			printf("prev_frame_bilinear_samples_validity_mask[%d]: %d != %d\n", i, (int)cuda_tmp_data.prev_frame_bilinear_samples_validity_mask[i], (int)opencl_tmp_data.prev_frame_bilinear_samples_validity_mask[i]);
		}
	}

	assert(cuda_tmp_data.spp.size() == opencl_tmp_data.spp.size());
	for(int i = 0; i < cuda_tmp_data.spp.size(); ++i)
	{
		if(cuda_tmp_data.spp[i] != opencl_tmp_data.spp[i])
		{
			printf("spp[%d]: %d != %d\n", i, (int)cuda_tmp_data.spp[i], (int)opencl_tmp_data.spp[i]);
		}
	}
#endif
	
#if 0
	CheckDiffFloat("min_max", cuda_tmp_data.features_min_max_buffer, opencl_tmp_data.features_min_max_buffer);
	CheckDiffFloat("features_buffer1", cuda_tmp_data.features_buffer1, opencl_tmp_data.features_buffer1);
	CheckDiffFloat("features_weights_buffer", cuda_tmp_data.features_weights_buffer, opencl_tmp_data.features_weights_buffer);

	CheckDiffFloat("noisefree_1spp", cuda_tmp_data.noisefree_1spp, opencl_tmp_data.noisefree_1spp);
	CheckDiffFloat("noisefree_1spp_accumulated", cuda_tmp_data.noisefree_1spp_accumulated, opencl_tmp_data.noisefree_1spp_accumulated);
	CheckDiffFloat("noisefree_1spp_acc_tonemapped", cuda_tmp_data.noisefree_1spp_acc_tonemapped, opencl_tmp_data.noisefree_1spp_acc_tonemapped);
#endif
	CheckDiffFloat("result", cuda_tmp_data.result, opencl_tmp_data.result);

	return 0;
}
