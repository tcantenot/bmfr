#include "opencl/bmfr.hpp"
#include "cuda/bmfr_cuda.hpp"

#include <assert.h>

// CUDA 8.0 on Visual Studio 2017
//  - Install CUDA 8.0
//  - Copy content of C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\extras\visual_studio_integration\MSBuildExtensions
//	  into C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\Common7\IDE\VC\VCTargets\BuildCustomizations
//  - Select Cuda 8.0 on the project right-click menu "Build Dependencies/Build Customizations..."
int main()
{
	TmpData cuda_tmp_data, opencl_tmp_data;
	bmfr_cuda(cuda_tmp_data);
    //bmfr_opencl();
	bmfr_c_opencl(opencl_tmp_data);
	
#if 0
	assert(cuda_tmp_data.normals.size() == opencl_tmp_data.normals.size());
	for(int i = 0; i < cuda_tmp_data.normals.size(); ++i)
	{
		if(cuda_tmp_data.normals[i] != opencl_tmp_data.normals[i])
		{
			printf("normals[%d]: %f != %f\n", i, cuda_tmp_data.normals[i], opencl_tmp_data.normals[i]);
		}
	}

	assert(cuda_tmp_data.positions.size() == opencl_tmp_data.positions.size());
	for(int i = 0; i < cuda_tmp_data.positions.size(); ++i)
	{
		if(cuda_tmp_data.positions[i] != opencl_tmp_data.positions[i])
		{
			printf("positions[%d]: %f != %f\n", i, cuda_tmp_data.positions[i], opencl_tmp_data.positions[i]);
		}
	}
#endif

#if 0
	assert(cuda_tmp_data.noisy_1spp.size() == opencl_tmp_data.noisy_1spp.size());
	for(int i = 0; i < cuda_tmp_data.noisy_1spp.size(); ++i)
	{
		if(cuda_tmp_data.noisy_1spp[i] != opencl_tmp_data.noisy_1spp[i])
		{
			printf("noisy_1spp[%d]: %f != %f\n", i, cuda_tmp_data.noisy_1spp[i], opencl_tmp_data.noisy_1spp[i]);
		}
	}
#endif

	assert(cuda_tmp_data.features_buffer.size() == opencl_tmp_data.features_buffer.size());
	for(int i = 0; i < cuda_tmp_data.features_buffer.size(); ++i)
	{
		if(fabs(cuda_tmp_data.features_buffer[i] - opencl_tmp_data.features_buffer[i]) > 1e-2f)
		{
			printf("features_buffer[%d]: %f != %f\n", i, cuda_tmp_data.features_buffer[i], opencl_tmp_data.features_buffer[i]);
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

	assert(cuda_tmp_data.features_weights_buffer.size() == opencl_tmp_data.features_weights_buffer.size());
	for(int i = 0; i < cuda_tmp_data.features_weights_buffer.size(); ++i)
	{
		if(cuda_tmp_data.features_weights_buffer[i] != opencl_tmp_data.features_weights_buffer[i])
		{
			printf("features_weights_buffer[%d]: %f != %f\n", i, cuda_tmp_data.features_weights_buffer[i], opencl_tmp_data.features_weights_buffer[i]);
		}
	}

	assert(cuda_tmp_data.features_min_max_buffer.size() == opencl_tmp_data.features_min_max_buffer.size());
	for(int i = 0; i < cuda_tmp_data.features_min_max_buffer.size(); ++i)
	{
		if(cuda_tmp_data.features_min_max_buffer[i] != opencl_tmp_data.features_min_max_buffer[i])
		{
			printf("features_min_max_buffer[%d]: %f != %f\n", i, cuda_tmp_data.features_min_max_buffer[i], opencl_tmp_data.features_min_max_buffer[i]);
		}
	}
	
	return 0;
}
