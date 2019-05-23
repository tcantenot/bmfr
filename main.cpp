#include "opencl/bmfr.hpp"
#include "cuda/bmfr_cuda.hpp"

#include <iostream>

#if 0
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