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

	const auto CheckDiffFloat = [](char const * name, std::vector<float> const & lhs, std::vector<float> const & rhs, bool bVerbose = false)
	{
		float maxDiff = 0, maxRelDiff = 0;
		int maxDiffIdx = 0, maxRelDiffIdx = 0;
		assert(lhs.size() == rhs.size());
		for(int i = 0; i < lhs.size(); ++i)
		{
			if(lhs[i] != rhs[i])
			{
				float lhs_fval = lhs[i] == 0.f ? abs(lhs[i]) : lhs[i];
				unsigned int lhs_uval;
				memcpy(&lhs_uval, &lhs_fval, sizeof(lhs_uval));

				float rhs_fval = rhs[i] == 0.f ? abs(rhs[i]) : rhs[i];
				unsigned int rhs_uval;
				memcpy(&rhs_uval, &rhs_fval, sizeof(rhs_uval));

				if(bVerbose)
				{
					unsigned int d = (rhs_uval > lhs_uval) ? (rhs_uval - lhs_uval) : (lhs_uval - rhs_uval);
					if(d > 1)
						printf("%s[%d]: %x != %x\n", name, i, lhs_uval, rhs_uval);
				}

				float diff = abs(lhs[i] - rhs[i]);
				float relDiff = abs(diff) / abs(rhs[i]);

				if(diff > maxDiff)
				{
					maxDiff = diff;
					maxDiffIdx = i;
				}

				if(relDiff > maxRelDiff)
				{
					maxRelDiff = relDiff;
					maxRelDiffIdx = i;
				}
			}
		}

		if(maxDiff > 0)
		{
			printf("%s[%d]: max diff = %.9g, %.9g != %.9g\n", name, maxDiffIdx, maxDiff, lhs[maxDiffIdx], rhs[maxDiffIdx]);
			printf("%s[%d]: max rel diff = %.9g, %.9g != %.9g\n", name, maxRelDiffIdx, maxRelDiff, lhs[maxRelDiffIdx], rhs[maxRelDiffIdx]);
		}
		else
		{
			printf("%s: identical!\n", name);
		}

		printf("\n");
	};
	
	printf("\n");

	CheckDiffFloat("normals", cuda_tmp_data.normals, opencl_tmp_data.normals);
	CheckDiffFloat("positions", cuda_tmp_data.positions, opencl_tmp_data.positions);
	CheckDiffFloat("noisy_1spp", cuda_tmp_data.noisy_1spp, opencl_tmp_data.noisy_1spp);
	CheckDiffFloat("min_max", cuda_tmp_data.features_min_max_buffer, opencl_tmp_data.features_min_max_buffer);
	CheckDiffFloat("features_buffer", cuda_tmp_data.features_buffer, opencl_tmp_data.features_buffer);
	CheckDiffFloat("features_weights_buffer", cuda_tmp_data.features_weights_buffer, opencl_tmp_data.features_weights_buffer);

	assert(cuda_tmp_data.spp.size() == opencl_tmp_data.spp.size());
	for(int i = 0; i < cuda_tmp_data.spp.size(); ++i)
	{
		if(cuda_tmp_data.spp[i] != opencl_tmp_data.spp[i])
		{
			printf("spp[%d]: %d != %d\n", i, (int)cuda_tmp_data.spp[i], (int)opencl_tmp_data.spp[i]);
		}
	}
	
	return 0;
}
