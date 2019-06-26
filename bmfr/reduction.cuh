#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "config.cuh"


// Unrolled parallel fmin reduction of 256 values
inline __device__ void parallel_reduction_min_256(
	float * __restrict__ result,
	float * __restrict__ shared_256,
	const unsigned int index
)
{
	if(index < 64)
		shared_256[index] = fmin(fmin(fmin(shared_256[index], shared_256[index + 64]), shared_256[index + 128]), shared_256[index + 192]);
	__syncthreads();

	if(index < 8)
		shared_256[index] = fmin(fmin(fmin(fmin(fmin(fmin(fmin(shared_256[index], shared_256[index + 8]),
			shared_256[index + 16]), shared_256[index + 24]), shared_256[index + 32]), shared_256[index + 40]),
			shared_256[index + 48]), shared_256[index + 56]);
	__syncthreads();

	if(index == 0)
	{
		*result = fmin(fmin(fmin(fmin(fmin(fmin(fmin(shared_256[0], shared_256[1]), shared_256[2]),
			shared_256[3]), shared_256[4]), shared_256[5]), shared_256[6]), shared_256[7]);
	}
	__syncthreads();
}

// Unrolled parallel fmax reduction of 256 values
inline __device__ void parallel_reduction_max_256(
	float * __restrict__ result,
	float * __restrict__ shared_256,
	const unsigned int index
)
{
	if(index < 64)
		shared_256[index] = fmax(fmax(fmax(shared_256[index], shared_256[index + 64]), shared_256[index + 128]), shared_256[index + 192]);
	__syncthreads();

	if(index < 8)
		shared_256[index] = fmax(fmax(fmax(fmax(fmax(fmax(fmax(shared_256[index], shared_256[index + 8]),
			shared_256[index + 16]), shared_256[index + 24]), shared_256[index + 32]), shared_256[index + 40]),
			shared_256[index + 48]), shared_256[index + 56]);
	__syncthreads();

	if(index == 0)
	{
		*result = fmax(fmax(fmax(fmax(fmax(fmax(fmax(shared_256[0], shared_256[1]), shared_256[2]),
			shared_256[3]), shared_256[4]), shared_256[5]), shared_256[6]), shared_256[7]);
	}
	__syncthreads();
}


// Unrolled parallel sum reduction of 256 values in shared memory
inline __device__ void parallel_reduction_sum_256(
	float * __restrict__ result,
	float * __restrict__ shared_256,
	const unsigned int index
)
{
	if(index < 64)
		shared_256[index] += shared_256[index + 64] + shared_256[index + 128] + shared_256[index + 192];
	__syncthreads();

	if(index < 8)
		shared_256[index] += shared_256[index + 8]  + shared_256[index + 16] + shared_256[index + 24] +
							 shared_256[index + 32] + shared_256[index + 40] + shared_256[index + 48] + shared_256[index + 56];
	__syncthreads();

	if(index == 0)
		*result = shared_256[0] + shared_256[1] + shared_256[2] + shared_256[3] +
				  shared_256[4] + shared_256[5] + shared_256[6] + shared_256[7];
	__syncthreads();
}

// Unrolled parallel sum reduction of 256 values (half-precision)
inline __device__ void parallel_reduction_sum_256(
	half * __restrict__ result,
	half * __restrict__ shared_256,
	const unsigned int index
)
{
	#if K_SUPPORT_HALF16_ARITHMETIC
	if(index < 64)
	{
		half2 tmp = __hadd2(__halves2half2(shared_256[index],		shared_256[index + 64]),
							__halves2half2(shared_256[index + 128], shared_256[index + 192])
					);

		shared_256[index] = __hadd(__high2half(tmp), __low2half(tmp));
	}
	__syncthreads();

	if(index < 8)
	{
		half2 tmp0 = __hadd2(__halves2half2(shared_256[index],		shared_256[index + 8]),
							 __halves2half2(shared_256[index + 16], shared_256[index + 24])
					 );

		half2 tmp1 = __hadd2(__halves2half2(shared_256[index + 32], shared_256[index + 40]),
							 __halves2half2(shared_256[index + 48], shared_256[index + 56])
					 );

		half2 tmp2 = __hadd2(tmp0, tmp1);

		shared_256[index] = __hadd(__high2half(tmp2), __low2half(tmp2));
	}
	__syncthreads();

	if(index == 0)
	{
		#if 0
		half2 tmp0 = __hadd2(__halves2half2(shared_256[0], shared_256[1]),
							 __halves2half2(shared_256[2], shared_256[3])
					 );

		half2 tmp1 = __hadd2(__halves2half2(shared_256[4], shared_256[5]),
							 __halves2half2(shared_256[6], shared_256[7])
					 );
		#else
		half2 tmp0 = __hadd2(*reinterpret_cast<half2*>(shared_256 + 0),
							 *reinterpret_cast<half2*>(shared_256 + 2)
					 );
		half2 tmp1 = __hadd2(*reinterpret_cast<half2*>(shared_256 + 4),
							 *reinterpret_cast<half2*>(shared_256 + 6)
					 );
		#endif

		half2 tmp2 = __hadd2(tmp0, tmp1);

		*result = __hadd(__high2half(tmp2), __low2half(tmp2));
	}
	__syncthreads();
	#else // K_SUPPORT_HALF16_ARITHMETIC
	if(index < 64)
	{
		shared_256[index] = __float2half(
								__half2float(shared_256[index])		  +
								__half2float(shared_256[index + 64])  + 
								__half2float(shared_256[index + 128]) +
								__half2float(shared_256[index + 192])
							);
	}
	__syncthreads();

	if(index < 8)
	{
		shared_256[index] = __float2half(
								__half2float(shared_256[index])		 +
								__half2float(shared_256[index + 8])  +
								__half2float(shared_256[index + 16]) +
								__half2float(shared_256[index + 24]) +
								__half2float(shared_256[index + 32]) +
								__half2float(shared_256[index + 40]) +
								__half2float(shared_256[index + 48]) +
								__half2float(shared_256[index + 56])
							);
	}
	__syncthreads();

	if(index == 0)
	{
		*result = __float2half(
					__half2float(shared_256[0]) +
					__half2float(shared_256[1]) +
					__half2float(shared_256[2]) +
					__half2float(shared_256[3]) +
					__half2float(shared_256[4]) +
					__half2float(shared_256[5]) +
					__half2float(shared_256[6]) +
					__half2float(shared_256[7])
				  );
	}
	__syncthreads();
	#endif // K_SUPPORT_HALF16_ARITHMETIC
}


// Unrolled parallel fmin reduction of 1024 values in shared memory
inline __device__ void parallel_reduction_min_1024(
	float * __restrict__ result,
	float * __restrict__ shared_1024,
	const unsigned int index
)
{
	if(index < 256)
	{
		shared_1024[index] = fmin(
			fmin(shared_1024[index], shared_1024[index + 256]),
			fmin(shared_1024[index + 512], shared_1024[index + 768])
		);
	}
	__syncthreads();

	parallel_reduction_min_256(result, shared_1024, index);
}

// Unrolled parallel fmax reduction of 1024 values in shared memory
inline __device__ void parallel_reduction_max_1024(
	float * __restrict__ result,
	float * __restrict__ shared_1024,
	const unsigned int index
)
{
	if(index < 256)
	{
		shared_1024[index] = fmax(
			fmax(shared_1024[index], shared_1024[index + 256]),
			fmax(shared_1024[index + 512], shared_1024[index + 768])
		);
	}
	__syncthreads();

	parallel_reduction_max_256(result, shared_1024, index);
}

// Unrolled parallel sum reduction of 1024 values in shared memory
inline __device__ void parallel_reduction_sum_1024(
	float * __restrict__ result,
	float * __restrict__ shared_1024,
	const unsigned int index
)
{
	if(index < 256)
	{
		shared_1024[index] += shared_1024[index + 256] + 
							  shared_1024[index + 512] +
							  shared_1024[index + 768];
	}
	__syncthreads();

	parallel_reduction_sum_256(result, shared_1024, index);
}

