#include "opencl/bmfr.hpp"
#include "cuda/bmfr_cuda.hpp"

#include <iostream>

// CUDA 8.0 on Visual Studio 2017
//  - Install CUDA 8.0
//  - Copy content of C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\extras\visual_studio_integration\MSBuildExtensions
//	  into C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\Common7\IDE\VC\VCTargets\BuildCustomizations
//  - Select Cuda 8.0 on the project right-click menu "Build Dependencies/Build Customizations..."
#if 1
int main()
{
#if 0
    try
    {
        return bmfr_opencl();
    }
	catch(...)
	{
		return 1;
	}
#else
	bmfr_cuda();
	char wait;
	std::cin >> wait;
	return 0;
#endif
}
#endif