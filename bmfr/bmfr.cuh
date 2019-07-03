#pragma once

#include "config.hpp"

#include "config.cuh"
#include "math.cuh"


// NOTE: if you want to use other than normal and world_position data you have to make
// it available in the first accumulation kernel and in the weighted sum kernel
#define NOT_SCALED_FEATURE_BUFFERS \
1.f,\
normal.x,\
normal.y,\
normal.z\

// The next features are not in the range from -1 to 1 so they are scaled to be from 0 to 1.
#if USE_SCALED_FEATURES
#define SCALED_FEATURE_BUFFERS \
,world_position.x,\
world_position.y,\
world_position.z,\
world_position.x*world_position.x,\
world_position.y*world_position.y,\
world_position.z*world_position.z
#else
#define SCALED_FEATURE_BUFFERS
#endif

#define FEATURE_BUFFERS NOT_SCALED_FEATURE_BUFFERS SCALED_FEATURE_BUFFERS


#define K_RESTRICT __restrict__

#ifdef __CUDACC__

////////////////////////////////////////////////////////////////////////////////

// Note: CUDA volatile qualifier
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#volatile-qualifier
// The compiler is free to optimize reads and writes to global or shared memory
// (for example, by caching global reads into registers or L1 cache) as long as
// it respects the memory ordering semantics of memory fence functions (Memory Fence Functions) and
// memory visibility semantics of synchronization functions (Synchronization Functions).
// These optimizations can be disabled using the volatile keyword: If a variable located in global or 
// shared memory is declared as volatile, the compiler assumes that its value can be changed or
// used at any time by another thread and therefore any reference to this variable compiles to 
// an actual memory read or write instruction.

// Note: Pointer aliasing and __restrict__
// https://devblogs.nvidia.com/cuda-pro-tip-optimize-pointer-aliasing/
// The __restrict__ keyword allows the compiler to know that two pointers do not alias.
// It also allows the use of the GPU read-only data cache, potentially accelerating data movement
// in a kernel.

// Note: Half-precision float
// https://devblogs.nvidia.com/mixed-precision-programming-cuda-8/
// Include header "cuda_fp16.h"

// TODO: add default defines when not compiling with NVRTC
// (might first divide "true" defines (constants or value dependent on the number of features) 
// and variables (values that depend on RT size, blend alpha values, ...)

// Threads synchronization /////////////////////////////////////////////////////

inline __device__ void SyncThreads()
{
	__syncthreads();
}

inline __device__ void GlobalMemFence()
{
	__syncthreads();
}

// Block offset contants ///////////////////////////////////////////////////////

// TODO: send as constant or define
#define BLOCK_EDGE_HALF (BLOCK_EDGE_LENGTH / 2)

// TODO: try to cycle through all offsets using Bayer matrix
#define BLOCK_OFFSETS_COUNT 16

__device__ __constant__ icvec2 BLOCK_OFFSETS_16[BLOCK_OFFSETS_COUNT] = {
	{  -7,  -7 },
	{   2,  -3 },
	{  -4,   7 },
	{   4,   0 },
	{  -5,  -4 },
	{   1,   6 },
	{   6,  -6 },
	{  -5,   0 },
	{   6,   7 },
	{  -4,  -8 },
	{   3,   3 },
	{  -1,  -1 },
	{   3,  -7 },
	{  -8,   6 },
	{   7,  -2 },
	{  -3,   2 }
};

__device__ __constant__ icvec2 BLOCK_OFFSETS_32[BLOCK_OFFSETS_COUNT] = {
	{ -14, -14 },
	{   4,  -6 },
	{  -8,  14 },
	{   8,   0 },
	{ -10,  -8 },
	{   2,  12 },
	{  12, -12 },
	{ -10,   0 },
	{  12,  14 },
	{  -8, -16 },
	{   6,   6 },
	{  -2,  -2 },
	{   6, -14 },
	{ -16,  12 },
	{  14,  -4 },
	{  -6,   4 }
};

__device__ __constant__ icvec2 BLOCK_OFFSETS_64[BLOCK_OFFSETS_COUNT] = {
	{ -28, -28 },
	{   8, -12 },
	{ -16,  28 },
	{  16,   0 },
	{ -20,  -4 },
	{   4,  24 },
	{  24, -24 },
	{ -20,   0 },
	{  24,  28 },
	{ -16, -32 },
	{  12,  12 },
	{  -4,  -4 },
	{  12, -28 },
	{ -32,  24 },
	{  28,  -8 },
	{ -12,   8 }
};

// R matrix indexing and operations ////////////////////////////////////////////

// TODO: change these defines either by macro that take parameters or inline functions
#if COMPRESSED_R
#define R_SIZE (R_EDGE * (R_EDGE + 1) / 2)
#define R_ROW_START (R_SIZE - (R_EDGE - y) * (R_EDGE - y + 1) / 2)
#define R_ACCESS (R_ROW_START + x - y)
// Reduces unused values in the begining of each row
// 00 01 02 03 04 05
// 11 12 13 14 15 22
// 23 24 25 33 34 35
// 44 45 55
#else
#define R_ACCESS (x * R_EDGE + y)
// Here - means unused value
// Note: "unused" values are still set to 0 so some operations can be done to
// every element in a row or column
//    0  1  2  3  4  5 x
// 0 00 01 02 03 04 05
// 1  - 11 12 13 14 15
// 2  -  - 22 23 24 25
// 3  -  -  - 33 34 35
// 4  -  -  -  - 44 45
// 5  -  -  -  -  - 55
// y
#endif

// TODO: if the function below do not work w/o volatile make them macros

inline __device__ vec3 load_r_mat(const cvec3 * r_mat, const int x, const int y)
{
   return r_mat[R_ACCESS];
}

inline __device__ void store_r_mat(/*volatile*/ cvec3* r_mat, const int x, const int y, vec3 value)
{
   r_mat[R_ACCESS] = *reinterpret_cast<cvec3*>(&value);
}

inline __device__ void store_r_mat_broadcast(/*volatile*/ cvec3 * r_mat, const int x, const int y, const float value)
{
	cvec3 v;
	v.x = value;
	v.y = value;
	v.z = value;
	r_mat[R_ACCESS] = v;
}

inline __device__ void store_r_mat_channel(/*volatile*/ cvec3 * r_mat, const int x, const int y, const int channel, const float value)
{
   if(channel == 0)
      r_mat[R_ACCESS].x = value;
   else if(channel == 1)
      r_mat[R_ACCESS].y = value;
   else // channel == 2
      r_mat[R_ACCESS].z = value;
}


// R matrix indexing and operations ////////////////////////////////////////////

// Random generator from here http://asgerhoedt.dk/?p=323
inline __device__ unsigned int ThrustHash(unsigned int seed)
{
	seed = (seed+0x7ed55d16) + (seed<<12);
	seed = (seed^0xc761c23c) ^ (seed>>19);
	seed = (seed+0x165667b1) + (seed<<5);
	seed = (seed+0xd3a2646c) ^ (seed<<9);
	seed = (seed+0xfd7046c5) + (seed<<3);
	seed = (seed^0xb55a4f09) ^ (seed>>16);
	return seed;
}

inline __device__ float ThrustRand01(unsigned int seed)
{
	return float(ThrustHash(seed)) / float(UINT_MAX);
}

inline __device__ unsigned int WangHash(unsigned int seed)
{
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}

inline __device__ float WangRand01(unsigned int seed)
{
	return float(WangHash(seed)) / float(UINT_MAX);
}

inline __device__ float SignedZeroMeanNoise(unsigned int seed)
{
	float noise01 = ThrustRand01(seed);
	//float noise01 = WangRand01(seed);
	return 2.f * noise01 - 1.f;
}


// Color space transformations /////////////////////////////////////////////////

inline __device__ vec3 RGB_to_YCoCg(vec3 rgb)
{
	return vec3(
		Dot(rgb, vec3(+1.f, +2.f, +1.f)),
		Dot(rgb, vec3(+2.f, +0.f, -2.f)),
		Dot(rgb, vec3(-1.f, +2.f, -1.f))
	);
}

inline __device__ vec3 YCoCg_to_RGB(vec3 YCoCg)
{
	return vec3(
		Dot(YCoCg, vec3(+0.25f, +0.25f, -0.25f)),
		Dot(YCoCg, vec3(+0.25f, +0.00f, +0.25f)),
		Dot(YCoCg, vec3(+0.25f, -0.25f, -0.25f))
	);
}


// Scaling functions ///////////////////////////////////////////////////////////

// TODO: try to scale in [-1, +1] to have the same interval for every feature
inline __device__ float scale(float value, float min, float max)
{
	if(Abs(max - min) > 1.0f)
	{
		return (value - min) / (max - min);
	}
	return value - min;
}

// Mirroring functions /////////////////////////////////////////////////////////

// Simple mirroring of image index if it is out of bounds.
// NOTE: Works only if index is less than one size out of bounds.
// NOTE: The mirroring duplicate borders: 3 2 1 0 | 0 1 2 3 | 3 2 1 0
inline __device__ int mirror(int index, int size)
{
	if(index < 0)
		index = Abs(index) - 1;
	else if(index >= size)
		index = 2 * size - index - 1;

	return index;
}

inline __device__ ivec2 mirror2(ivec2 index, ivec2 size)
{
	index.x = mirror(index.x, size.x);
	index.y = mirror(index.y, size.y);
	return index;
}

// Conversion functions ////////////////////////////////////////////////////////

inline __device__ half FloatToHalf(float x)
{
	return __float2half(x);
}

inline __device__ float HalfToFloat(half x)
{
	return __half2float(x);
}

inline __device__ int FloatToIntRd(float x)
{
	return __float2int_rd(x);
}

inline __device__ ivec2 FloatToIntRd(vec2 v)
{
	return ivec2(__float2int_rd(v.x), __float2int_rd(v.y));
}

inline __device__ ivec3 FloatToIntRd(vec3 v)
{
	return ivec3(__float2int_rd(v.x), __float2int_rd(v.y), __float2int_rd(v.z));
}

inline __device__ ivec4 FloatToIntRd(vec4 v)
{
	return ivec4(__float2int_rd(v.x), __float2int_rd(v.y), __float2int_rd(v.z), __float2int_rd(v.w));
}

inline __device__ unsigned char convert_uchar_sat_rte(float x)
{
	unsigned int u = __float2uint_rn(x);
	return static_cast<unsigned char>(Min(Max(u, 0u), 255u));
}

// Load/store vec3 functions /////////////////////////////////////////////////

inline __device__ vec3 load_float3(float const * K_RESTRICT buffer, unsigned int  index)
{
	#if OPTIMIZE_LOAD_STORE
	return (*reinterpret_cast<vec4 const *>(buffer + index * 3)).xyz();
	#else
	return vec3(buffer[index * 3 + 0], buffer[index * 3 + 1], buffer[index * 3 + 2]);
	#endif
}	

inline __device__ void store_float3(float * K_RESTRICT buffer, unsigned int  index, vec3 const & value)
{
	#if OPTIMIZE_LOAD_STORE
	*reinterpret_cast<vec3 *>(buffer + index * 3) = value;
	#else
	buffer[index * 3 + 0] = value.x;
	buffer[index * 3 + 1] = value.y;
	buffer[index * 3 + 2] = value.z;
	#endif
}

#if USE_HALF_PRECISION_IN_FEATURES_DATA
inline __device__ float load_feature(half const * buffer, unsigned int index)
#else
inline __device__ float load_feature(float const * buffer, unsigned int index)
#endif
{
	#if USE_HALF_PRECISION_IN_FEATURES_DATA
	return HalfToFloat(buffer[index]);
	#else
	return buffer[index];
	#endif
}

#if USE_HALF_PRECISION_IN_FEATURES_DATA
inline __device__ void store_feature(half * buffer, unsigned int index, float value)
#else
inline __device__ void store_feature(float * buffer, unsigned int index, float value)
#endif
{
	#if USE_HALF_PRECISION_IN_FEATURES_DATA
	buffer[index] = FloatToHalf(value);
	#else
	buffer[index] = value;
	#endif
}
#endif // __CUDACC__


// Accumulate noisy 1spp color kernel //////////////////////////////////////////

struct AccumulateNoisyDataKernelParams
{
	unsigned int sizeX;
	unsigned int sizeY;
	unsigned int fitterBlockSize;
	unsigned int worksetWithMarginBlockCountX;
	unsigned int frameNumber;
};

extern "C" void run_accumulate_noisy_data(
	AccumulateNoisyDataKernelParams const & params,
	dim3 const & grid_size,
	dim3 const & block_size,
	vec2 * K_RESTRICT out_prev_frame_pixel,			// [out] Previous frame pixel coordinates (after reprojection)
	unsigned char* K_RESTRICT accept_bools,			// [out] Validity mask of bilinear samples in previous frame (after reprojection)
	const float * K_RESTRICT current_normals,		// [in]  Current  (world) normals
	const float * K_RESTRICT previous_normals,		// [in]  Previous (world) normals
	const float * K_RESTRICT current_positions,		// [in]  Current  world positions
	const float * K_RESTRICT previous_positions,	// [in]  Previous world positions
	const float * K_RESTRICT frame_noisy_1spp,		// [in]  Frame noisy 1spp color buffer
		  float * K_RESTRICT current_noisy,			// [out] Current  accumulated noisy 1spp color
	const float * K_RESTRICT previous_noisy,		// [in]  Previous accumulated noisy 1spp color
	const unsigned char * K_RESTRICT previous_spp,	// [in]  Previous number of samples accumulated (for CMA)
		  unsigned char * K_RESTRICT current_spp,	// [out] Current  number of samples accumulated (for CMA)
	#if USE_HALF_PRECISION_IN_FEATURES_DATA
	half * K_RESTRICT features_data,				// [out] Features buffer (half-precision)
	#else
	float * K_RESTRICT features_data,				// [out] Features buffer (single-precision)
	#endif
	const mat4x4 prev_frame_camera_matrix,			// [in]  ViewProj matrix of previous frame
	const vec2 pixel_offset
);

// Fitter kernel ///////////////////////////////////////////////////////////////

struct FitterKernelParams
{
	unsigned int fitterBlockSize;
	unsigned int kernelLocalSize;
	unsigned int worksetWithMarginBlockCountX;
	unsigned int frameNumber;
};

extern "C" void run_fitter(
	dim3 const & grid_size,
	dim3 const & block_size,
	FitterKernelParams const & params,
	float * K_RESTRICT weights,					// [out] Features weights
	float * K_RESTRICT mins_maxs,				// [out] Min and max of features values per block (world_positions)
	#if USE_HALF_PRECISION_IN_FEATURES_DATA
	half * K_RESTRICT features_buffer			// [out] Features buffer (half-precision)
	#else
	float * K_RESTRICT features_buffer			// [out] Features buffer (single-precision)
	#endif
);

// Weighted sum kernel /////////////////////////////////////////////////////////
// -> outputs the noise-free 1spp color estimate

struct WeightedSumKernelParams
{
	unsigned int sizeX;
	unsigned int sizeY;
	unsigned int fitterBlockSize;
	unsigned int worksetWithMarginBlockCountX;
	unsigned int frameNumber;
};

extern "C" void run_weighted_sum(
	dim3 const & grid_size,
	dim3 const & block_size,
	WeightedSumKernelParams const & params,
	const float * K_RESTRICT weights,			// [in]	 Features weights computed by the fitter kernel
	const float * K_RESTRICT mins_maxs,			// [in]  Min and max of features values per block (world_positions)
		  float * K_RESTRICT output,			// [out] Noise-free color estimate
	const float * K_RESTRICT current_normals,	// [in]  Current (world) normals
	const float * K_RESTRICT current_positions	// [in]  Current world positions
);


// Accumulate filtered data kernel /////////////////////////////////////////////
// -> outputs the noise-free accumulated color estimate + a tonemapped version w/ albedo

struct AccumulateFilteredDataKernelParams
{
	unsigned int sizeX;
	unsigned int sizeY;
	unsigned int frameNumber;
};

extern "C" void run_accumulate_filtered_data(
	dim3 const & grid_size,
	dim3 const & block_size,
	AccumulateFilteredDataKernelParams const & params,
	const float * K_RESTRICT filtered_frame,			// [in]  Noise free color estimate (computed as the weighted sum of the features)
	const vec2 * K_RESTRICT in_prev_frame_pixel,		// [in]  Previous frame pixel coordinates (after reprojection)
	const unsigned char * K_RESTRICT accept_bools,		// [in]  Validity mask of bilinear samples in previous frame (after reprojection)
	const float * K_RESTRICT albedo_buffer,				// [in]  Albedo buffer of the current frame (non-noisy)
		  float * K_RESTRICT tone_mapped_frame,			// [out] Accumulated and tonemapped noise-free color estimate
	const unsigned char* K_RESTRICT current_spp,		// [in]	 Current number of samples accumulated (for CMA)
	const float * K_RESTRICT accumulated_prev_frame,	// [in]  Previous frame noise-free accumulated color estimate 
		  float * K_RESTRICT accumulated_frame			// [out] Current frame noise-free accumulated color estimate
);


// TAA kernel //////////////////////////////////////////////////////////////////

struct TAAKernelParams
{
	unsigned int sizeX;
	unsigned int sizeY;
	unsigned int frameNumber;
};

extern "C" void run_taa(
	dim3 const & grid_size,
	dim3 const & block_size,
	TAAKernelParams const & params,
	const vec2 * K_RESTRICT in_prev_frame_pixel,	// [in]  Previous frame pixel coordinates (after reprojection)
	const float * K_RESTRICT new_frame,				// [in]	 Current frame color buffer
		  float * K_RESTRICT result_frame,			// [out] Antialiased frame color buffer
	const float * K_RESTRICT prev_frame				// [in]  Previous frame color buffer
);

////////////////////////////////////////////////////////////////////////////////

inline __device__ vec3 HeatMap(float value01)
{
    const int N = 9;
    const vec4 HeatMapColorsAndLevels[N] =
    {
        vec4(0.0f, 0.0f, 0.0f, 0.0f / float(N-1)),	// black (0, 0, 0)
        vec4(0.0f, 0.0f, 1.0f, 1.0f / float(N-1)),	// blue (0, 0, 1)
        vec4(0.0f, 1.0f, 1.0f, 2.0f / float(N-1)),	// cyan (0, 1, 1)
        vec4(0.0f, 1.0f, 0.0f, 3.0f / float(N-1)),	// green (0, 1, 0)
        vec4(1.0f, 1.0f, 0.0f, 4.0f / float(N-1)),	// yellow (1, 1, 0)
        vec4(1.0f, 0.5f, 0.0f, 5.0f / float(N-1)),	// orange (1, 0.5, 0)
        vec4(1.0f, 0.0f, 0.0f, 6.0f / float(N-1)),	// red (1, 0, 0)
        vec4(1.0f, 0.0f, 1.0f, 7.0f / float(N-1)),	// magenta (1, 0, 1)
        vec4(1.0f, 1.0f, 1.0f, 8.0f / float(N-1))	// white (1, 1, 1)
    };

    vec3 heatmap = HeatMapColorsAndLevels[0].xyz();
    for(int i = 1; i < N; ++i)
    {
        float currLvl = HeatMapColorsAndLevels[i].w;
        float prevLvl = HeatMapColorsAndLevels[i-1].w;
        vec3  currCol = HeatMapColorsAndLevels[i].xyz();
        vec3  prevCol = HeatMapColorsAndLevels[i-1].xyz();
        if(value01 <= currLvl)
        {
            heatmap = Lerp(currCol, prevCol, (currLvl - value01) / (currLvl - prevLvl));
            break;
        }
    }

    return heatmap;
}

////////////////////////////////////////////////////////////////////////////////

extern "C" void run_cuda_hello();
