#include "opencl/bmfr.hpp"
#include "cuda/bmfr_cuda.hpp"

// CUDA 8.0 on Visual Studio 2017
//  - Install CUDA 8.0
//  - Copy content of C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\extras\visual_studio_integration\MSBuildExtensions
//	  into C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\Common7\IDE\VC\VCTargets\BuildCustomizations
//  - Select Cuda 8.0 on the project right-click menu "Build Dependencies/Build Customizations..."
int main()
{
	//bmfr_cuda();
    //bmfr_opencl();
	bmfr_c_opencl();
	return 0;
}
