#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "config.hpp"

////////////////////////////////////////////////////////////////////////////////
// TODO replace: Placeholders

template <typename T>
inline __device__ T Min(T lhs, T rhs) {	return lhs < rhs ? lhs : rhs; }

template <typename T>
inline __device__ T Max(T lhs, T rhs) {	return lhs < rhs ? rhs : lhs; }

template <typename T>
inline __device__ T Clamp(T x, T a, T b) { return Max(Min(x, b), a); }

template <typename T>
inline __device__ T Saturate(T x) { return Clamp(x, T(0), T(1)); }

template <typename T>
inline __device__ T Sqrt(T x) { return sqrt(x); }

template <typename T>
struct tvec2
{
	T x, y;

	__device__ explicit tvec2(T v = T(0)): x(v), y(v) { }
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
	__device__ explicit tvec3(T v = T(0)) { x = v; y = v; z = v; }
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
inline __device__ tvec3<T> operator*(tvec3<T> const & v, T x) {	return tvec3<T>(x * v.x, x * v.y, x * v.z); }

template <typename T>
inline __device__ tvec3<T> operator/(tvec3<T> const & v, T x) {	return tvec3<T>(v.x / x, v.y / x, v.z / x); }


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

template <typename T>
inline __device__ T Lerp(T a, T b, T t)
{
	return (T(1) - t) * a + t * b;
}

template <typename T>
inline __device__ tvec3<T> Lerp(tvec3<T> const & a, tvec3<T> const & b, tvec3<T> const & t)
{
	return tvec3<T>(Lerp(a.x, b.x, t.x), Lerp(a.y, b.y, t.y), Lerp(a.z, b.z, t.z));
}

template <typename T>
inline __device__ tvec3<T> Lerp(tvec3<T> const & a, tvec3<T> const & b, T t)
{
	return tvec3<T>(Lerp(a.x, b.x, t), Lerp(a.y, b.y, t), Lerp(a.z, b.z, t));
}

using ivec3 = tvec3<int>;
using vec3 = tvec3<float>;

template <typename T>
inline __device__ T Dot(tvec3<T> const & lhs, tvec3<T> const & rhs) { return lhs.x*rhs.x + lhs.y*rhs.y + lhs.z*rhs.z; }

template <typename T>
struct tvec4
{
	T x, y, z, w;

	__device__ explicit tvec4(T v = T(0)): x(v), y(v), z(v), w(v) { }
	__device__ tvec4(T xx, T yy, T zz, T ww): x(xx), y(yy), z(zz), w(ww) { }
	__device__ tvec4(tvec3<T> const & v, T ww): x(v.x), y(v.y), z(v.z), w(ww) { }
	__device__ tvec4 operator+(tvec4 const & o) { return tvec4(x + o.x, y + o.y, z + o.z, w + o.w); }
	__device__ tvec4 operator-(tvec4 const & o) { return tvec4(x - o.x, y - o.y, z - o.z, w - o.w); }

	__device__ T operator[](int i) const { return *((&x) + i); }

	__device__ tvec3<T> xyz() const { return tvec3<T>(x, y, z); }
};

using ivec4 = tvec4<int>;
using vec4 = tvec4<float>;

template <typename T>
struct tmat4x4
{
	tvec4<T> m[4]; // Column major
	__device__ tvec4<T> row(int i) const { return tvec4<T>(m[0][i], m[1][i], m[2][i], m[3][i]); }
	__device__ tvec4<T> const & operator[](int i) const { return m[i]; }
};

using mat4x4 = tmat4x4<float>;

template <typename T>
inline __device__ T Dot(tvec4<T> const & lhs, tvec4<T> const & rhs) { return lhs.x*rhs.x + lhs.y*rhs.y + lhs.z*rhs.z + lhs.w*rhs.w; }

template <typename T>
inline __device__ T Abs(T x) { return x < 0 ? -x : x; }

#define C_FLT_MAX INFINITY

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

extern "C" void run_test(
	dim3 const & grid_size,
	dim3 const & block_size,
	float * data
);

// Accumulate noisy 1spp color kernel //////////////////////////////////////////

extern "C" void run_accumulate_noisy_data(
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
	const vec2 pixel_offset,
	const int frame_number							// [in]  Current frame number
);

// Fitter kernel ///////////////////////////////////////////////////////////////

extern "C" void run_fitter(
	dim3 const & grid_size,
	dim3 const & block_size,
	float * K_RESTRICT weights,					// [out] Features weights
	float * K_RESTRICT mins_maxs,				// [out] Min and max of features values per block (world_positions)
	#if USE_HALF_PRECISION_IN_FEATURES_DATA
	half * K_RESTRICT features_buffer,			// [out] Features buffer (half-precision)
	#else
	float * K_RESTRICT features_buffer,			// [out] Features buffer (single-precision)
	#endif
	const int frame_number						// [in]  Current frame number
);

// Weighted sum kernel /////////////////////////////////////////////////////////
// -> outputs the noise-free 1spp color estimate

extern "C" void run_weighted_sum(
	dim3 const & grid_size,
	dim3 const & block_size,
	const float * K_RESTRICT weights,			// [in]	 Features weights computed by the fitter kernel
	const float * K_RESTRICT mins_maxs,			// [in]  Min and max of features values per block (world_positions)
		  float * K_RESTRICT output,			// [out] Noise-free color estimate
	const float * K_RESTRICT current_normals,	// [in]  Current (world) normals
	const float * K_RESTRICT current_positions,	// [in]  Current world positions
	const float * K_RESTRICT current_noisy,		// [in]  Current noisy 1spp color (only used for debugging)
	const int frame_number						// [in]  Current frame number
);

// Accumulate filtered data kernel /////////////////////////////////////////////
// -> outputs the noise-free accumulated color estimate + a tonemapped version w/ albedo

extern "C" void run_accumulate_filtered_data(
	dim3 const & grid_size,
	dim3 const & block_size,
	const float * K_RESTRICT filtered_frame,			// [in]  Noise free color estimate (computed as the weighted sum of the features)
	const vec2 * K_RESTRICT in_prev_frame_pixel,		// [in]  Previous frame pixel coordinates (after reprojection)
	const unsigned char * K_RESTRICT accept_bools,		// [in]  Validity mask of bilinear samples in previous frame (after reprojection)
	const float * K_RESTRICT albedo_buffer,				// [in]  Albedo buffer of the current frame (non-noisy)
		  float * K_RESTRICT tone_mapped_frame,			// [out] Accumulated and tonemapped noise-free color estimate
	const unsigned char* K_RESTRICT current_spp,		// [in]	 Current number of samples accumulated (for CMA)
	const float * K_RESTRICT accumulated_prev_frame,	// [in]  Previous frame noise-free accumulated color estimate 
		  float * K_RESTRICT accumulated_frame,			// [out] Current frame noise-free accumulated color estimate
	const int frame_number								// [in]  Current frame number
);

// TAA kernel //////////////////////////////////////////////////////////////////

extern "C" void run_taa(
	dim3 const & grid_size,
	dim3 const & block_size,
	const vec2 * K_RESTRICT in_prev_frame_pixel,	// [in]  Previous frame pixel coordinates (after reprojection)
	const float * K_RESTRICT new_frame,				// [in]	 Current frame color buffer
		  float * K_RESTRICT result_frame,			// [out] Antialiased frame color buffer
	const float * K_RESTRICT prev_frame,			// [in]  Previous frame color buffer
	const int frame_number							// [in]  Current frame number
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
