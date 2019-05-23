#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

////////////////////////////////////////////////////////////////////////////////
// TODO replace: Placeholders

template <typename T>
inline __device__ T Min(T lhs, T rhs) {	return lhs < rhs ? lhs : rhs; }

template <typename T>
inline __device__ T Max(T lhs, T rhs) {	return lhs < rhs ? rhs : lhs; }

template <typename T>
inline __device__ T Clamp(T x, T a, T b) { return Max(Min(x, b), a); }

template <typename T>
inline __device__ T Sqrt(T x) { return sqrt(x); }

template <typename T>
struct tvec2
{
	T x, y;

	__device__ tvec2(T v = T(0)): x(v), y(v) { }
	__device__ tvec2(T xx, T yy): x(xx), y(yy) { }
	template <typename U>
	__device__ tvec2(tvec2<U> const & o): x(o.x), y(o.y) { }
	__device__ tvec2(float2 const & o): x(o.x), y(o.y) { }
	__device__ tvec2 operator+(tvec2 const & o) { return tvec2(x + o.x, y + o.y); }
	__device__ tvec2 operator-(tvec2 const & o) { return tvec2(x - o.x, y - o.y); }
	__device__ tvec2 operator*(tvec2 const & o) { return tvec2(x * o.x, y * o.y); }

	__device__ tvec2 const & operator-=(tvec2<T> const & v) { x -= v.x; y -= v.y; return *this; }

	__device__ tvec2 const & operator+=(T v) { x += v; y += v; return *this; }
	__device__ tvec2 const & operator*=(T v) { x *= v; y *= v; return *this; }
	__device__ tvec2 const & operator/=(T v) { x /= v; y /= v; return *this; }
};

template <typename T>
inline __device__ tvec2<T> operator+(tvec2<T> const & l, tvec2<T> const & r) { return tvec2<T>(l.x + r.x, l.y + r.y); }

template <typename T>
inline __device__ tvec2<T> operator-(tvec2<T> const & l, tvec2<T> const & r) { return tvec2<T>(l.x - r.x, l.y - r.y); }

template <typename T>
inline __device__ tvec2<T> operator-(T x, tvec2<T> const & v) {	return tvec2<T>(x - v.x, x - v.y); }

template <typename T>
inline __device__ tvec2<T> operator-(tvec2<T> const & v, T x) {	return tvec2<T>(v.x - x, v.y - x); }

template <typename T>
inline __device__ tvec2<T> operator+(tvec2<T> const & v, T x) {	return tvec2<T>(v.x + x, v.y + x); }


using ivec2 = tvec2<int>;
using vec2 = tvec2<float>;


template <typename T>
struct tcvec3
{
	T x, y, z;
};

using cvec3 = tcvec3<float>;

template <typename T>
struct tvec3 : public tcvec3<T>
{
	__device__ tvec3(T v = 0) { x = v; y = v; z = v; }
	__device__ tvec3(T xx, T yy, T zz) { x = xx; y = yy; z = zz; }
	__device__ tvec3(tcvec3<T> const & o) { x = o.x; y = o.y; z = o.z; }
	__device__ tvec3 & operator=(tvec3 const & o) { x = o.x; y = o.y; z = o.z; return *this; }
	__device__ tvec3 operator+(tvec3 const & o) { return tvec3(x + o.x, y + o.y, z + o.z); }
	__device__ tvec3 operator-(tvec3 const & o) { return tvec3(x - o.x, y - o.y, z - o.z); }
	__device__ tvec3 operator*(tvec3 const & o) { return tvec3(x * o.x, y * o.y, z * o.z); }

	__device__ tvec3 const & operator+=(tvec3 const & o) { x += o.x; y += o.y; z += o.z; return *this; }

	__device__ tvec3 const & operator/=(T v) { x /= v; y /= v; z /= v; return *this; }

	__device__ tvec3 operator/(tvec3<T> const & v) { return tvec3<T>(x / v.x, y / v.y, z / v.z); }
};

template <typename T>
inline __device__ tvec3<T> operator*(T x, tvec3<T> const & v) {	return tvec3<T>(x * v.x, x * v.y, x * v.z); }

template <typename T>
inline __device__ tvec3<T> Min(tvec3<T> const & l, tvec3<T> const & r)
{
	return tvec3<T>(Min(l.x, r.x), Min(l.y, r.y), Min(l.z, r.z));
}

template <typename T>
inline __device__ tvec3<T> Max(tvec3<T> const & l, tvec3<T> const & r)
{
	return tvec3<T>(Max(l.x, r.x), Max(l.y, r.y), Max(l.z, r.z));
}

template <typename T>
inline __device__ tvec3<T> Clamp(tvec3<T> const & v, tvec3<T> const & a, tvec3<T> const & b)
{
	return tvec3<T>(Clamp(v.x, a.x, b.x), Clamp(v.y, a.y, b.y), Clamp(v.z, a.z, b.z));
}

template <typename T>
inline __device__ tvec3<T> Pow(tvec3<T> const & v, T p)
{
	return tvec3<T>(pow(v.x, p), pow(v.y, p), pow(v.z, p));
}

using ivec3 = tvec3<int>;
using vec3 = tvec3<float>;

template <typename T>
inline __device__ T Dot(tvec3<T> const & lhs, tvec3<T> const & rhs) { return lhs.x*rhs.x + lhs.y*rhs.y + lhs.z*rhs.z; }

template <typename T>
struct tvec4
{
	T x, y, z, w;

	__device__ tvec4(T v = 0): x(v), y(v), z(v), w(v) { }
	__device__ tvec4(T xx, T yy, T zz, T ww): x(xx), y(yy), z(zz), w(ww) { }
	__device__ tvec4(tvec3<T> const & v, T ww): x(v.x), y(v.y), z(v.z), w(ww) { }
	__device__ tvec4 operator+(tvec4 const & o) { return tvec4(x + o.x, y + o.y, z + o.z, w + o.w); }
	__device__ tvec4 operator-(tvec4 const & o) { return tvec4(x - o.x, y - o.y, z - o.z, w - o.w); }

	__device__ T operator[](int i) const { return *(&x + i); }

	__device__ tvec3<T> xyz() const { return tvec3<T>(x, y, z); }
};

using ivec4 = tvec4<int>;
using vec4 = tvec4<float>;

template <typename T>
struct tmat4x4
{
	tvec4<T> m[4]; // Column major
	__device__ tvec4<T> row(int i) const { return tvec4<T>(m[0][i], m[1][i], m[2][i], m[3][i]); }
};

using mat4x4 = tmat4x4<float>;

template <typename T>
inline __device__ T Dot(tvec4<T> const & lhs, tvec4<T> const & rhs) { return lhs.x*rhs.x + lhs.y*rhs.y + lhs.z*rhs.z + lhs.w*rhs.w; }

template <typename T>
inline __device__ T Abs(T x) { return x < 0 ? -x : x; }

#define C_FLT_MAX INFINITY


#define BLOCK_EDGE_LENGTH 32
#define BLOCK_PIXELS (BLOCK_EDGE_LENGTH * BLOCK_EDGE_LENGTH)


// ### Edit these defines if you want to experiment different parameters ###
// The amount of noise added to feature buffers to cancel sigularities
#define NOISE_AMOUNT 1e-2

// The amount of new frame used in accumulated frame (1.f would mean no accumulation).
#define BLEND_ALPHA 0.2f
#define SECOND_BLEND_ALPHA 0.1f
#define TAA_BLEND_ALPHA 0.2f

// NOTE: if you want to use other than normal and world_position data you have to make
// it available in the first accumulation kernel and in the weighted sum kernel
#define NOT_SCALED_FEATURE_BUFFERS \
1.f,\
normal.x,\
normal.y,\
normal.z,\
// The next features are not in the range from -1 to 1 so they are scaled to be from 0 to 1.
#define SCALED_FEATURE_BUFFERS \
world_position.x,\
world_position.y,\
world_position.z,\
world_position.x*world_position.x,\
world_position.y*world_position.y,\
world_position.z*world_position.z

#define BUFFER_COUNT 13
#define FEATURES_NOT_SCALED 4
#define FEATURES_SCALED 6
#define IMAGE_WIDTH 1280
#define IMAGE_HEIGHT 720

// Rounds image sizes up to next multiple of BLOCK_EDGE_LENGTH
#define WORKSET_WIDTH  (BLOCK_EDGE_LENGTH * ((IMAGE_WIDTH  + BLOCK_EDGE_LENGTH - 1) / BLOCK_EDGE_LENGTH))
#define WORKSET_HEIGHT (BLOCK_EDGE_LENGTH * ((IMAGE_HEIGHT + BLOCK_EDGE_LENGTH - 1) / BLOCK_EDGE_LENGTH))

#define FEATURE_BUFFERS NOT_SCALED_FEATURE_BUFFERS SCALED_FEATURE_BUFFERS

// These local sizes are used with 2D kernels which do not require spesific local size
// (Global sizes are always a multiple of 32)
#define LOCAL_WIDTH 8
#define LOCAL_HEIGHT 8


// We need a margin of one block (BLOCK_EDGE_LENGTH) because we are offsetting the blocks
// at most one block to the left or the right to avoid blocky artifacts.
// So most of the time, we have 2 partially filled blocks and BLOCK_COUNT_X-1 full blocks on one row

// Example: image of 8x12 with blocks of 4x1

// No offset: 2x3 = 6 blocks
// |....|....|
// |....|....|
// |....|....|

// With some offset (here (-2, 0)): 3x3 = 9 blocks
// |--..|....|..--|
// |--..|....|..--|
// |--..|....|..--|

// Workset with margin width
#define WORKSET_WITH_MARGINS_WIDTH (WORKSET_WIDTH + BLOCK_EDGE_LENGTH)

// Workset with margin height
#define WORKSET_WITH_MARGINS_HEIGHT (WORKSET_HEIGHT + BLOCK_EDGE_LENGTH)

// Number of blocks in X dim in the workset with margin
#define WORKSET_WITH_MARGIN_BLOCK_COUNT_X (WORKSET_WITH_MARGINS_WIDTH  / BLOCK_EDGE_LENGTH)

// Number of blocks in Y dim in the workset with margin
#define WORKSET_WITH_MARGIN_BLOCK_COUNT_Y (WORKSET_WITH_MARGINS_HEIGHT / BLOCK_EDGE_LENGTH)

// Number of block in the workset with margin
#define WORKSET_WITH_MARGIN_BLOCK_COUNT (WORKSET_WITH_MARGIN_BLOCK_COUNT_X * WORKSET_WITH_MARGIN_BLOCK_COUNT_Y)

#define R_EDGE (BUFFER_COUNT - 2)

#define POSITION_LIMIT_SQUARED 0.010000f
#define NORMAL_LIMIT_SQUARED 1.000000f

// ### Edit these defines to change optimizations for your target hardware ###
// If 1 uses ~half local memory space for R, but computing indexes is more complicated
#define COMPRESSED_R 1

// If 1 stores tmp_data to private memory when it is loaded for dot product calculation
#define CACHE_TMP_DATA 1

// If 1 features_data buffer is in half precision for faster load and store.
// NOTE: if world position values are greater than 256 this cannot be used because
// 256*256 is infinity in half-precision
#define USE_HALF_PRECISION_IN_FEATURES_DATA 0

// 256 is the maximum local size on AMD GCN
// Synchronization within 32x32=1024 block requires unrolling four times
#define LOCAL_SIZE 256



// Accumulate noisy 1spp color kernel //////////////////////////////////////////

extern "C" void run_accumulate_noisy_data(
	dim3 const & grid_size,
	dim3 const & block_size,
	vec2 * __restrict__ out_prev_frame_pixel,			// [out] Previous frame pixel coordinates (after reprojection)
	unsigned char* __restrict__ accept_bools,			// [out] Validity mask of bilinear samples in previous frame (after reprojection)
	const float * __restrict__ current_normals,			// [in]  Current  (world) normals
	const float * __restrict__ previous_normals,		// [in]  Previous (world) normals
	const float * __restrict__ current_positions,		// [in]  Current  world positions
	const float * __restrict__ previous_positions,		// [in]  Previous world positions
		  float * __restrict__ current_noisy,			// [out] Current  noisy 1spp color
	const float * __restrict__ previous_noisy,			// [in]  Previous noisy 1spp color
	const unsigned char * __restrict__ previous_spp,	// [in]  Previous number of samples accumulated (for CMA)
		  unsigned char * __restrict__ current_spp,		// [out] Current  number of samples accumulated (for CMA)
	#if USE_HALF_PRECISION_IN_FEATURES_DATA
	half * __restrict__ features_data,					// [out] Features buffer (half-precision)
	#else
	float * __restrict__ features_data,					// [out] Features buffer (single-precision)
	#endif
      const mat4x4 prev_frame_camera_matrix,			// [in]  ViewProj matrix of previous frame
      const vec2 pixel_offset,
      const int frame_number							// [in]  Current frame number
);

// Fitter kernel ///////////////////////////////////////////////////////////////

extern "C" void run_fitter(
	dim3 const & grid_size,
	dim3 const & block_size,
	float * __restrict__ weights,					// [out] Features weights
	float * __restrict__ mins_maxs,					// [out] Min and max of features values per block (world_positions)
	#if USE_HALF_PRECISION_IN_FEATURES_DATA
	half * __restrict__ features_buffer,			// [out] Features buffer (half-precision)
	#else
	float * __restrict__ features_buffer,			// [out] Features buffer (single-precision)
	#endif
	const int frame_number							// [in]  Current frame number
);

// Weighted sum kernel /////////////////////////////////////////////////////////
// -> outputs the noise-free 1spp color estimate

extern "C" void run_weighted_sum(
	dim3 const & grid_size,
	dim3 const & block_size,
	const float * __restrict__ weights,				// [in]	 Features weights computed by the fitter kernel
	const float * __restrict__ mins_maxs,			// [in]  Min and max of features values per block (world_positions)
		  float * __restrict__ output,				// [out] Noise-free color estimate
	const float * __restrict__ current_normals,		// [in]  Current (world) normals
	const float * __restrict__ current_positions,	// [in]  Current world positions
	const float * __restrict__ current_noisy,		// [in]  Current noisy 1spp color (only used for debugging)
	const int frame_number							// [in]  Current frame number
);

// Accumulate filtered data kernel /////////////////////////////////////////////
// -> outputs the noise-free accumulated color estimate + a tonemapped version w/ albedo

extern "C" void run_accumulate_filtered_data(
	dim3 const & grid_size,
	dim3 const & block_size,
	const float * __restrict__ filtered_frame,			// [in]  Noise free color estimate (computed as the weighted sum of the features)
	const vec2 * __restrict__ in_prev_frame_pixel,		// [in]  Previous frame pixel coordinates (after reprojection)
	const unsigned char * __restrict__ accept_bools,	// [in]  Validity mask of bilinear samples in previous frame (after reprojection)
	const float * __restrict__ albedo_buffer,			// [in]  Albedo buffer of the current frame (non-noisy)
		  float * __restrict__ tone_mapped_frame,		// [out] Accumulated and tonemapped noise-free color estimate
	const unsigned char* __restrict__ current_spp,		// [in]	 Current number of samples accumulated (for CMA)
	const float * __restrict__ accumulated_prev_frame,	// [in]  Previous frame noise-free accumulated color estimate 
		  float * __restrict__ accumulated_frame,		// [out] Current frame noise-free accumulated color estimate
	const int frame_number								// [in]  Current frame number
);

// TAA kernel //////////////////////////////////////////////////////////////////

extern "C" void run_taa(
	dim3 const & grid_size,
	dim3 const & block_size,
	const vec2 * __restrict__ in_prev_frame_pixel,	// [in]  Previous frame pixel coordinates (after reprojection)
	const float * __restrict__ new_frame,			// [in]	 Current frame color buffer
		  float * __restrict__ result_frame,		// [out] Antialiased frame color buffer
	const float * __restrict__ prev_frame,			// [in]  Previous frame color buffer
	const int frame_number							// [in]  Current frame number
);

////////////////////////////////////////////////////////////////////////////////

extern "C" void run_cuda_hello();
