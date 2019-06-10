#include "bmfr.cuh"

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

// Parallel reductions /////////////////////////////////////////////////////////

// Unrolled parallel sum reduction of 256 values
// TODO: unused start_index...
inline __device__ void parallel_reduction_sum_256(float * K_RESTRICT result, float * K_RESTRICT pr_data_256, const int start_index)
{
	const int id = threadIdx.x;

	if(id < 64)
		pr_data_256[id] += pr_data_256[id + 64] + pr_data_256[id + 128] + pr_data_256[id + 192];
	SyncThreads();

	if(id < 8)
		pr_data_256[id] += pr_data_256[id + 8]  + pr_data_256[id + 16] + pr_data_256[id + 24] +
						   pr_data_256[id + 32] + pr_data_256[id + 40] + pr_data_256[id + 48] + pr_data_256[id + 56];
	SyncThreads();

	if(id == 0)
	{
		*result = pr_data_256[0] + pr_data_256[1] + pr_data_256[2] + pr_data_256[3] +
				  pr_data_256[4] + pr_data_256[5] + pr_data_256[6] + pr_data_256[7];
	}
	SyncThreads();
}

// TODO: replace by Min4
// Unrolled parallel min reduction of 256 values
inline __device__ void parallel_reduction_min_256(float * K_RESTRICT result, float * K_RESTRICT pr_data_256)
{
	const int id = threadIdx.x;

	if(id < 64)
		pr_data_256[id] = Min(Min(Min(pr_data_256[id], pr_data_256[id + 64]), pr_data_256[id + 128]), pr_data_256[id + 192]);
	SyncThreads();

	if(id < 8)
		pr_data_256[id] = Min(Min(Min(Min(Min(Min(Min(pr_data_256[id], pr_data_256[id + 8]),
			pr_data_256[id + 16]), pr_data_256[id + 24]), pr_data_256[id + 32]), pr_data_256[id + 40]),
			pr_data_256[id + 48]), pr_data_256[id + 56]);
	SyncThreads();

	if(id == 0)
	{
		*result = Min(Min(Min(Min(Min(Min(Min(pr_data_256[0], pr_data_256[1]), pr_data_256[2]),
			pr_data_256[3]), pr_data_256[4]), pr_data_256[5]), pr_data_256[6]), pr_data_256[7]);
	}
	SyncThreads();
}

// TODO: replace by Max4
// Unrolled parallel max reduction of 256 values
inline __device__ void parallel_reduction_max_256(float * K_RESTRICT result, float * K_RESTRICT pr_data_256)
{
   const int id = threadIdx.x;

	if(id < 64)
		pr_data_256[id] = Max(Max(Max(pr_data_256[id], pr_data_256[id + 64]), pr_data_256[id + 128]), pr_data_256[id + 192]);
	SyncThreads();

	if(id < 8)
		pr_data_256[id] = Max(Max(Max(Max(Max(Max(Max(pr_data_256[id], pr_data_256[id + 8]),
			pr_data_256[id + 16]), pr_data_256[id + 24]), pr_data_256[id + 32]), pr_data_256[id + 40]),
			pr_data_256[id + 48]), pr_data_256[id + 56]);
	SyncThreads();

	if(id == 0)
	{
		*result = Max(Max(Max(Max(Max(Max(Max(pr_data_256[0], pr_data_256[1]), pr_data_256[2]),
			pr_data_256[3]), pr_data_256[4]), pr_data_256[5]), pr_data_256[6]), pr_data_256[7]);
	}
	SyncThreads();
}


// Block offset contants ///////////////////////////////////////////////////////

// TODO: send as constant or define
#define BLOCK_EDGE_HALF (BLOCK_EDGE_LENGTH / 2)

// TODO: try to cycle through all offsets using Bayer matrix
#define BLOCK_OFFSETS_COUNT 16
__device__ __constant__ float2 BLOCK_OFFSETS[BLOCK_OFFSETS_COUNT] = {
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

#if USE_HALF_PRECISION_IN_FEATURES_DATA
inline __device__ half FloatToHalf(float x)
{
	return __float2half(x);
}

inline __device__ float HalfToFloat(half x)
{
	return __half2float(x);
}
#endif

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

inline __device__ void store_float3(float * K_RESTRICT buffer, unsigned int  index, vec3 value)
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

#endif


// Accumulate noisy 1spp color kernel //////////////////////////////////////////

__global__ void accumulate_noisy_data(
	AccumulateNoisyDataKernelParams params,
	vec2 * K_RESTRICT out_prev_frame_pixel,			// [out] Previous frame pixel coordinates (after reprojection)
	unsigned char* K_RESTRICT accept_bools,			// [out] Validity mask of bilinear samples in previous frame (after reprojection)
	const float * K_RESTRICT current_normals,		// [in]  Current  (world) normals
	const float * K_RESTRICT previous_normals,		// [in]  Previous (world) normals
	const float * K_RESTRICT current_positions,		// [in]  Current  world positions
	const float * K_RESTRICT previous_positions,	// [in]  Previous world positions
	const float * K_RESTRICT frame_noisy_1spp,		// [in]  Frame noisy 1spp color buffer
		  float * K_RESTRICT current_noisy,			// [out] Current  noisy 1spp color
	const float * K_RESTRICT previous_noisy,		// [in]  Previous noisy 1spp color
	const unsigned char * K_RESTRICT previous_spp,	// [in]  Previous number of samples accumulated (for CMA)
		  unsigned char * K_RESTRICT current_spp,	// [out] Current  number of samples accumulated (for CMA)
	#if USE_HALF_PRECISION_IN_FEATURES_DATA
	half * K_RESTRICT features_data,				// [out] Features buffer (half-precision)
	#else
	float * K_RESTRICT features_data,				// [out] Features buffer (single-precision)
	#endif
	const mat4x4 prev_frame_camera_matrix,			// [in]  ViewProj matrix of previous frame
	const vec2 pixel_offset
)
{
	const ivec2 gid = ivec2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

	const int w = params.sizeX;
	const int h = params.sizeY;

	// Mirror indexed of the input. x and y are always less than one size out of
	// bounds if image dimensions are bigger than BLOCK_EDGE_LENGTH
	// BLOCK_EDGE_HALF = half block size (32/2 -> 16)
	const ivec2 pixel_without_mirror = gid - BLOCK_EDGE_HALF + BLOCK_OFFSETS[params.frameNumber % BLOCK_OFFSETS_COUNT];

	// Pixel coordinates in [0, w-1]x[0, h-1]
	const ivec2 pixel = mirror2(pixel_without_mirror, ivec2(w, h));

	// Linear pixel index in image in [0, w*h-1]
	const int linear_pixel = pixel.y * w + pixel.x;
   
	// Current frame noisy color (1spp)
	// [Section 3.1]
	// The input for the real-time reconstruction filter is a 1 spp path-traced frame and its accompanying feature buffers.
	// The 1 spp frames are generated by using a rasterizer for producing the primary rays and feature buffers.
	// We use mipmapped textures in albedo. Next, we do so-called next event estimation: we trace one shadow ray towards
	// a random point in one random light source and then continue path tracing by sending one secondary ray to a random direction.
	// Namely, we use multiple importance sampling [Veach and Guibas 1995].
	// The direction of the secondary ray is decided based on importance sampling. We also trace a second shadow ray from the
	// intersection point of the secondary ray.
	// Consequently, the 1spp pixel input has one rasterized primary ray (non-noisy), one ray-traced secondary ray and two ray-traced shadow rays.
	const vec3 current_color = load_float3(frame_noisy_1spp, linear_pixel);

	// Current frame world position
	const vec4 world_position = vec4(load_float3(current_positions, linear_pixel), 1.0f);

	// Current frame (world) normal
	const vec3 normal = load_float3(current_normals, linear_pixel);


	// Previous frame pixel coordinates in [0, w-1]x[0, h-1]
	// Default is the same pixel coordinates (no movement)
	vec2 prev_frame_pixel_f = vec2(pixel.x, pixel.y);

	// Bit mask telling which previous frame (bilinear) samples are valid under reprojection into current frame
	unsigned char store_accept = 0x00;

	// Blending factor with history buffer
	// Blend_alpha 1.f means that only current frame color is used. The value is changed if sample from previous frame can be used
	float blend_alpha = 1.f;
	vec3 previous_color = vec3(0.f, 0.f, 0.f);

	float sample_spp = 0.f;
	if(params.frameNumber > 0)
	{
		// Project current world position into previous frame with the previous ViewProj matrix

		// Matrix multiplication and normalization to 0..1
		vec2 prev_frame_uv;

		// TODO: send matrix transposed
		prev_frame_uv.x = Dot(prev_frame_camera_matrix.row(0), world_position); // Transform x
		prev_frame_uv.y = Dot(prev_frame_camera_matrix.row(1), world_position); // Transform y
		// No need for z-buffer in accumulation of the noisy data
		// -> might be useful if we use it to detect disocclusion
		// --> compare previous z (store previous frame Z-buffer) with prev_frame_pixel_uv.z
		//prev_frame_uv.z = Dot(prev_frame_camera_matrix.row(2), world_position); // Transform z
		prev_frame_uv /= Dot(prev_frame_camera_matrix.row(3), world_position);

		prev_frame_uv = prev_frame_uv * vec2(0.5f) + vec2(0.5f);

		// Compute the pixel coordinates in the previous frame (in [0, w-1]x[0, h-1])
		prev_frame_pixel_f = prev_frame_uv * vec2(w, h);

		// Apply offset (TODO: what offset??? seems to always be 0.5... Maybe TAA/ray subpixel offsets -> send 1-pixel_offset.y)
		// TODO: try to remove this
		prev_frame_pixel_f -= vec2(pixel_offset.x, 1 - pixel_offset.y);

		// Convert into integer pixel coordinates (round down)
		const ivec2 prev_frame_pixel_i = FloatToIntRd(prev_frame_pixel_f);

		// Compute bilinear weights (for bilinear sampling)
		// TODO: implement bicubic Catmull-Rom (for sharpness)? => would need to perform more fetches and store more "validity bits" in mask
		const ivec2 offsets[4] = { ivec2(0, 0), ivec2(1, 0), ivec2(0, 1), ivec2(1, 1) };

		const vec2 prev_pixel_fract = prev_frame_pixel_f - vec2(prev_frame_pixel_i);
		const vec2 one_minus_prev_pixel_fract = 1.f - prev_pixel_fract;

		float weights[4];
		weights[0] = one_minus_prev_pixel_fract.x * one_minus_prev_pixel_fract.y;
		weights[1] = prev_pixel_fract.x           * one_minus_prev_pixel_fract.y;
		weights[2] = one_minus_prev_pixel_fract.x * prev_pixel_fract.y;
		weights[3] = prev_pixel_fract.x           * prev_pixel_fract.y;
		float total_weight = 0.f;

		// Bilinear sampling
		for(int i = 0; i < 4; ++i)
		{
			ivec2 sample_location = prev_frame_pixel_i + offsets[i];

			// Check if previous frame color can be used based on its screen location
			if(sample_location.x >= 0 && sample_location.y >= 0 &&
			   sample_location.x < w && sample_location.y < h
			)
			{
				const int linear_sample_location = sample_location.y * w + sample_location.x;

				// Fetch previous frame world position
				vec3 prev_world_position = load_float3(previous_positions, linear_sample_location);

				// Compute world distance squared
				vec3 position_difference = prev_world_position - world_position.xyz();
				float position_distance_squared = Dot(position_difference, position_difference);

				// World position distance discard
				if(position_distance_squared < float(POSITION_LIMIT_SQUARED))
				{
					// Fetch previous frame normal
					vec3 prev_normal = load_float3(previous_normals, linear_sample_location);

					// Distance of the normals
					// TODO: could use some other distance metric (e.g. angle), but we use hard
					// experimentally found threshold -> means that the metric doesn't matter.
					vec3 normal_difference = prev_normal - normal;
					float normal_distance_squared = Dot(normal_difference, normal_difference);

					if(normal_distance_squared < float(NORMAL_LIMIT_SQUARED))
					{
						// Pixel passes all tests so store it to "validity bitmask"
						store_accept |= 1 << i;

						// Accumulate number of samples
						sample_spp += weights[i] * float(previous_spp[linear_sample_location]);

						// Accumulate previous noisy 1spp color
						previous_color += weights[i] * load_float3(previous_noisy, linear_sample_location);

						// Acumulate weights
						total_weight += weights[i];
					}
				}
			}
		}

		if(total_weight > 0.f)
		{
			previous_color /= total_weight;
			sample_spp /= total_weight;

			// Cumulative Moving Average (CMA)
			// CMA_n = (x_1 + x_2 + ... + x_n) / n
			// <=> (x_1 + x_2 + ... + x_n) = n * CMA_n
			// CMA_(n+1) = (x_1 + x_2 + ... + x_n + x_(n+1)) / (n + 1)
			//			  = (n * CMA_n + x_(n+1)) / (n + 1)
			//			  = n/(n+1) * CMA_n + 1/(n+1) * x_(n+1)
			//			  = (n+1-1)/(n+1) * CMA_n + 1/(n+1) * x_(n+1)
			//			  = (1 - 1/(n+1)) * CMA_n + 1/(n+1) * x_(n+1)
			//			  = lerp(CMA_n, x_(n+1), 1/(n+1))
			blend_alpha = 1.f / (sample_spp + 1.f);

			// Blend_alpha is dymically decided so that the result is average
			// of all samples (cumulative moving average) until the cap defined by
			// BLEND_ALPHA is reached (exponential moving average: EMA_(n+1) = (1 - a) * EMA_n + a * x_(n+1) = lerp(EMA_n, x_(n+1), a))

			// [Section 3.2]
			// We start by computing a cumulative moving average of the samples, 
			// and use the exponential moving average only after the cumulative moving average weight
			// of the new sample would be less than BLEND_ALPHA (e.g 20%).
			// The use of regular average on the first frames and after occlusions makes sure that
			// the first samples do not get an excessively high weight, and limiting the weight to a minimum
			// of BLEND_APLHA (e.g 20%) makes sure that the aged data fades away.
			blend_alpha = Max(blend_alpha, BLEND_ALPHA);
		}
	}

	// Store new spp
	unsigned char new_spp = 1;
	if(blend_alpha < 1.f) // alpha = 1.f means we ignore history
	{
		// Note: we accumulate at most 255 samples for the cumulative moving average (which is more than enough because of
		// the threshold BLEND_ALPHA that switch to exponential moving average).
		// E.g: BLEND_ALPHA = 0.2 = 1 / (n + 1) <=> n = (1 - 0.2) / 0.2 = 4 => above 4 samples for a pixel, we switch to
		// exponential moving average with alpha = 20%
		// n = 255 <=> alpha = 1.0 / (255 + 1) = 0.0039
		
		// TODO: store "validity mask" along the "spp" in 8 bits: 4-bit validity mask | 4-bit spp
		// 4-bit spp <=> max n = 2^4-1 = 15 <=> min alpha = 1.0 / (5 + 1) = 0.0625

		new_spp = (sample_spp > 254.f) ? 255 : convert_uchar_sat_rte(sample_spp) + 1;
	}

	vec3 new_color = blend_alpha * current_color + (1.f - blend_alpha) * previous_color; // Lerp(previous_color, current_color, blend_alpha);

	// The set of feature buffers used in the fitting
	float features[BUFFER_COUNT] =
	{
		FEATURE_BUFFERS, // expands to 1.f, normal.x, ..., world_position.x, ..., world_position.x * world_position.x, ...
		new_color.x,
		new_color.y,
		new_color.z
	};

	const unsigned int x_block = gid.x / BLOCK_EDGE_LENGTH; // Block coordinate x
	const unsigned int y_block = gid.y / BLOCK_EDGE_LENGTH; // Block coordinate y
	const unsigned int x_in_block = gid.x % BLOCK_EDGE_LENGTH; // Thread coordinate x inside block in [0, BLOCK_EDGE_LENGTH-1]
	const unsigned int y_in_block = gid.y % BLOCK_EDGE_LENGTH; // Thread coordinate y inside block in [0, BLOCK_EDGE_LENGTH-1]

	const unsigned int features_base_offset = x_in_block + y_in_block * BLOCK_EDGE_LENGTH +
		x_block * BLOCK_PIXELS * BUFFER_COUNT +
		y_block * params.worksetWithMarginBlockCountX *
		BLOCK_PIXELS * BUFFER_COUNT;
	
	for(unsigned int feature_num = 0; feature_num < BUFFER_COUNT; ++feature_num)
	{
		// Index in feature buffer (data are concatenated)
		// | Block 0 feature 0 | Block 0 feature 0 | ... | Block 0 feature M | ... | Block 1 feature 0 | ... | Block N feature 0 | ... | Block N feature M |
		const unsigned int location_in_data = features_base_offset + feature_num * BLOCK_PIXELS;

		float feature = features[feature_num];

		// TODO: remove -> useless
		if(isnan(feature))
			feature = 0.0f;

		// TODO: remove -> useless when features will be normalized
		#if USE_HALF_PRECISION_IN_FEATURES_DATA
		feature = Clamp(feature, -65504.f, 65504.f);
		#endif

		store_feature(features_data, location_in_data, feature);
	}

	// The kernel works on a workset of size WORKSET_WITH_MARGINS_WIDTH x WORKSET_WITH_MARGINS_HEIGHT
	// -> the extra block margin is used to handle the offsets applied to reduce the block artifacts.
	// [Section 3.5]: "To aid the reduction of blockiness, BMFR processes each frame over a grid of non-overlapping
	//	blocks which is displaced with random offsets. These offsets prevent the artifacts that would arise from reusing
	// same block positions on a static scene with a static camera."
	// --> Only the pixels inside the image (after applying the offsets) should write to the output data that
	// have the same size of the input image
	if(pixel_without_mirror.x >= 0 && pixel_without_mirror.x < w &&
	   pixel_without_mirror.y >= 0 && pixel_without_mirror.y < h
	)
	{
		store_float3(current_noisy, linear_pixel, new_color); // Accumulated noisy 1spp
		out_prev_frame_pixel[linear_pixel] = prev_frame_pixel_f; // Previous frame pixel coordinates (to sample history)
		accept_bools[linear_pixel] = store_accept; // "Previous frame bilinear samples validity" bitmask
		current_spp[linear_pixel] = new_spp; // Store current number of samples accumulated (for CMA)

		// Kernel debug: stored in current_noisy buffer
		#if 0
		vec3 debug = vec3(0.0f);
		//debug = vec3(prev_frame_uv.x, prev_frame_uv.y, 0);
		//debug = vec3(blend_alpha);
		debug = HeatMap(Saturate(float(new_spp) / 255.f));
		//debug = vec3(float(store_accept > 0));
		//debug = vec3(float(store_accept == ((1 << 4) - 1)));
		store_float3(current_noisy, linear_pixel, debug);
		#endif
	}
}

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
		  float * K_RESTRICT current_noisy,			// [out] Current  noisy 1spp color
	const float * K_RESTRICT previous_noisy,		// [in]  Previous noisy 1spp color
	const unsigned char * K_RESTRICT previous_spp,	// [in]  Previous number of samples accumulated (for CMA)
		  unsigned char * K_RESTRICT current_spp,	// [out] Current  number of samples accumulated (for CMA)
	#if USE_HALF_PRECISION_IN_FEATURES_DATA
	half * K_RESTRICT features_data,				// [out] Features buffer (half-precision)
	#else
	float * K_RESTRICT features_data,				// [out] Features buffer (single-precision)
	#endif
	const mat4x4 prev_frame_camera_matrix,			// [in]  ViewProj matrix of previous frame
	const vec2 pixel_offset
)
{
	accumulate_noisy_data<<<grid_size, block_size>>>(
		params,
		out_prev_frame_pixel,
		accept_bools,
		current_normals,
		previous_normals,
		current_positions,
		previous_positions,
		frame_noisy_1spp,
		current_noisy,
		previous_noisy,
		previous_spp,
		current_spp,
		features_data,
		prev_frame_camera_matrix,
		pixel_offset
	);
}

// TODO: add an extra pass to normalize features and copy them to a single buffer
// -> normalize world_position only once and then compute world_position*world_position
// --> avoid parallel reduction for higher order features
// --> avoid output min/max that is used to scale the scaled features in the kernel "weighted_sum"
//	   (there seems to be double scaling: once in fitter and once weighted_sum...)
//
// TODO: add a constant for the "real" number of features which equal to BUFFER_COUNT - 3
//
// OR directly generate a normalized world_position by dividing by the bbox of the scene or some value + saturate
// -> allow to skip parallel min/max reductions altogether

// Fitter kernel ///////////////////////////////////////////////////////////////

// Block size: (256, 1, 1)
__global__ void fitter(
	FitterKernelParams params,
	float * K_RESTRICT weights,					// [out] Features weights
	float * K_RESTRICT mins_maxs,				// [out] Min and max of features values per block (world_positions)
	#if USE_HALF_PRECISION_IN_FEATURES_DATA
	half * K_RESTRICT features_buffer			// [out] Features buffer (half-precision)
	#else
	float * K_RESTRICT features_buffer			// [out] Features buffer (single-precision)
	#endif
)
{
	// Notes:
	//  LOCAL_SIZE = 256
	//	BLOCK_PIXELS = 32 * 32
	
	// TODO: send as define for cpp side
	#if COMPRESSED_R
    //const auto r_size = ((buffer_count - 2) * (buffer_count - 1) / 2) * sizeof(cl_float3);
	#define R_SHARED_DATA_SIZE ((BUFFER_COUNT - 2) * (BUFFER_COUNT - 1) / 2)
	#else
    //const auto r_size = (buffer_count - 2) * (buffer_count - 2) * sizeof(cl_float3);
	#define R_SHARED_DATA_SIZE ((BUFFER_COUNT - 2) * (BUFFER_COUNT - 2))
	#endif

	__shared__ float pr_shared_data[LOCAL_SIZE];		// Shared memory used to perform parallel reduction (max, min, sum)
	__shared__ float u_vec_sdata[BLOCK_PIXELS];			// Shared memory used to store the 'u' vectors
	__shared__ cvec3 r_mat_sdata[R_SHARED_DATA_SIZE];	// Shared memory used to store the R matrices of the QR factorization (vec3 -> one per color channel)
	__shared__ float u_length_squared;					// Shared memory variable that holds the 'u' vector square length
	__shared__ float dotProd;							// Shared memory variable that holds the dot product of...
	__shared__ float block_min;							// Shared memory variable that holds the result of the parallel min reduction
	__shared__ float block_max;							// Shared memory variable that holds the result of the parallel max reduction
	__shared__ float vec_length;						// Shared memory variable that holds the vec length			

	float * pr_data_256 = &pr_shared_data[0];
	float * u_vec = &u_vec_sdata[0];
	cvec3 * r_mat = &r_mat_sdata[0];

	const int groupId = blockIdx.x;
	const int threadId = threadIdx.x; // in [0, 255]

	const unsigned int blockIndexX = groupId % params.worksetWithMarginBlockCountX;
	const unsigned int blockIndexY = groupId / params.worksetWithMarginBlockCountX;
	const unsigned int linearBlockIndex = blockIndexY * params.worksetWithMarginBlockCountX + blockIndexX;
	const unsigned int threadFeaturesBuffersOffset = linearBlockIndex * BUFFER_COUNT * BLOCK_PIXELS + threadId;

	// Scales world positions to 0..1 in a block
	const int feature_to_scale_beg_idx = FEATURES_NOT_SCALED;
	const int feature_to_scale_end_idx = BUFFER_COUNT - 3;
	for(int featureIndex = feature_to_scale_beg_idx; featureIndex < feature_to_scale_end_idx; ++featureIndex)
	{
		const unsigned int baseFeatureOffset = featureIndex * BLOCK_PIXELS + threadFeaturesBuffersOffset;

		// Find maximum and minimum of the whole block
		float tmp_max = -C_FLT_MAX;
		float tmp_min = +C_FLT_MAX;
	  
		// Manual unrolling for parallel reduction as the block contains 1024 (32x32) work items and
		// the reduction operates on 256 elements (group size)
		// -> Compute the min and max of N values (N = 1024/256 = 4)
		const int N = BLOCK_PIXELS / LOCAL_SIZE;
		for(int subVector = 0; subVector < N; ++subVector)
		{
			const unsigned int featureOffset = subVector * LOCAL_SIZE + baseFeatureOffset;
			float value = load_feature(features_buffer, featureOffset);
			tmp_max = Max(value, tmp_max);
			tmp_min = Min(value, tmp_min);
		}

		// Parallel min reduction
		pr_data_256[threadId] = tmp_min;
		SyncThreads();
		parallel_reduction_min_256(&block_min, pr_data_256);

		// Parallel max reduction
		pr_data_256[threadId] = tmp_max;
		SyncThreads();
		parallel_reduction_max_256(&block_max, pr_data_256);

		// Output the min and max features values per block of 32x32 pixels (only output 256 values because of manual unrolling of 4)
		if(threadId == 0)
		{
			const int index = (groupId * FEATURES_SCALED + (featureIndex - feature_to_scale_beg_idx)) * 2;
			mins_maxs[index + 0] = block_min;
			mins_maxs[index + 1] = block_max;
		}
		SyncThreads(); // TODO: this thread synchronization may not be useful

		// Scale feature and replace value in features buffer
		for(int subVector = 0; subVector < BLOCK_PIXELS / LOCAL_SIZE; ++subVector)
		{
			const unsigned int featureOffset = subVector * LOCAL_SIZE + baseFeatureOffset;
			float scaled_value = scale(load_feature(features_buffer, featureOffset), block_min, block_max);
			store_feature(features_buffer, featureOffset, scaled_value);
		}
	}

	const unsigned int baseSeed = params.frameNumber * BUFFER_COUNT * BLOCK_PIXELS + threadId;
	
	// Non square matrices require processing every column.
	// Otherwise result is OKish, but R is not upper triangular matrix
	const int limit = (BUFFER_COUNT == BLOCK_PIXELS) ? BUFFER_COUNT - 1 : BUFFER_COUNT;

	// Compute R
	for(int col = 0; col < limit; col++)
	{
		// Note: the last 3 features values are the 3 channels of the color (not used for the regression)
		int col_limited = Min(col, BUFFER_COUNT - 3);

		// Load new column into memory
		const int featureIndex = col;

		//TODO: reduced scope of this as it is recreated further below
		const unsigned int baseFeatureOffset = featureIndex * BLOCK_PIXELS + threadFeaturesBuffersOffset;
		
		float tmp_sum_value = 0.f;

		// Manual unrolling for parallel reduction as the block contains 1024 (32x32) work items and
		// the reduction operates on 256 elements (group size)
		// -> Compute the sum of N values (N = 1024/256 = 4)
		for(int subVector = 0; subVector < BLOCK_PIXELS / LOCAL_SIZE; ++subVector)
		{
			const unsigned int featureOffset = subVector * LOCAL_SIZE + baseFeatureOffset;

			// Load feature
			float tmp = load_feature(features_buffer, featureOffset);

			// Store the feature in shared memory
			const int index = subVector * LOCAL_SIZE + threadId;
			u_vec[index] = tmp;

			if(index >= col_limited + 1)
			{
				tmp_sum_value += tmp * tmp;
			}
		}
		SyncThreads();

		// Find length of vector in A's column with reduction sum function
		pr_data_256[threadId] = tmp_sum_value;
		SyncThreads();
		parallel_reduction_sum_256(&vec_length, pr_data_256, col_limited + 1);

		// NOTE: GCN Opencl compiler can do some optimization with this because if
		// initially wanted col_limited is used to select wich work-item runs which branch
		// it is slower. However using col produces the same result.
		float r_value;
		if(threadId < col)
		{
			// Copy u_vec value
			r_value = u_vec[threadId];
		}
		else if(threadId == col)
		{
			u_length_squared = vec_length;
			vec_length = Sqrt(vec_length + u_vec[col_limited] * u_vec[col_limited]);
			u_vec[col_limited] -= vec_length;
			u_length_squared += u_vec[col_limited] * u_vec[col_limited];

			// (u_length_squared is now updated length squared)
			r_value = vec_length;
		}
		else if(threadId > col) //Could have "&& threadId <  R_EDGE" but this is little bit faster
		{
			// Last values on every column are zeros
			r_value = 0.0f;
		}

		int id_limited = Min(threadId, BUFFER_COUNT - 3);
		if(col < BUFFER_COUNT - 3)
			store_r_mat_broadcast(r_mat, col_limited, id_limited, r_value);
		else
			store_r_mat_channel(r_mat, col_limited, id_limited, col - BUFFER_COUNT + 3, r_value);
		SyncThreads();

		// Transform further columns of A
		// NOTE: three last columns are three color channels of noisy data. However,
		// they all need to be transfomed as they were column indexed (buffers - 3)
		for(int featureIndex = col_limited+1; featureIndex < BUFFER_COUNT; ++featureIndex)
		{
			const unsigned int baseFeatureOffset = featureIndex * BLOCK_PIXELS + threadFeaturesBuffersOffset;
			const unsigned int baseFeatureSeed   = featureIndex * BLOCK_PIXELS + baseSeed;

			// Starts by computing dot product with reduction sum function
			#if CACHE_TMP_DATA
			// No need to load features_buffer twice because each work-item first copies value for
			// dot product computation and then modifies the same value
			float tmp_data_private_cache[BLOCK_PIXELS / LOCAL_SIZE];
			#endif

			float tmp_sum_value = 0.f;
			for(int subVector = 0; subVector < BLOCK_PIXELS / LOCAL_SIZE; ++subVector)
			{
				const unsigned int featureOffset = subVector * LOCAL_SIZE + baseFeatureOffset;
				const int index = subVector * LOCAL_SIZE + threadId;
				if(index >= col_limited)
				{
					// Load feature
					float tmp = load_feature(features_buffer, featureOffset);

					// [Section 3.4] - Stochastic regularization
					// To handle rank-deficiency in the T matrix, add zero-mean noise to the input buffers
					// (the first time values are loaded), which makes them linearly independent.
					// Note: does not add noise to constant buffer (column 0) and noisy image data (last 3 columns).
					if(col == 0 && featureIndex < BUFFER_COUNT - 3)
					{
						const int seed = subVector * LOCAL_SIZE + baseFeatureSeed;
						tmp += NOISE_AMOUNT * SignedZeroMeanNoise(seed);
					}

					#if CACHE_TMP_DATA
					tmp_data_private_cache[subVector] = tmp;
					#endif
					tmp_sum_value += tmp * u_vec[index];
				}
			}

			pr_data_256[threadId] = tmp_sum_value;
			SyncThreads();
			parallel_reduction_sum_256(&dotProd, pr_data_256, col_limited);

			const float dotFactor = 2.0f * dotProd / u_length_squared;

			// Manual unrolling as the block contains 1024 (32x32) work items and we operate on 256 elements (group size)
			// -> Compute the sum of N values (N = 1024/256 = 4)
			for(int subVector = 0; subVector < BLOCK_PIXELS / LOCAL_SIZE; ++subVector)
			{
				const unsigned int featureOffset = subVector * LOCAL_SIZE + baseFeatureOffset;
				const int index = subVector * LOCAL_SIZE + threadId;
				if(index >= col_limited)
				{
					#if CACHE_TMP_DATA
					float store_value = tmp_data_private_cache[subVector];
					#else
					float store_value = load_feature(features_buffer, featureOffset);
					const int seed = sub_vector * LOCAL_SIZE + baseFeatureSeed;
					store_value += NOISE_AMOUNT * SignedZeroMeanNoise(seed);
					#endif
					store_value -= dotFactor * u_vec[index];
					store_feature(features_buffer, featureOffset, store_value);
				}
			}
			GlobalMemFence();
		}
	}

	// Back substitution
	__shared__ cvec3 divider; // Shared memory variable that holds the divider

	// R_EDGE = buffer_count - 2 (= number of features + 3 (noisy color spp buffer) - 2)
	// R is a (M + 1)x(M + 1) matrix, with M the number of features (here equal to buffer_count - 3)
	// which gives us R_EDGE = M + 1 = buffer_count - 3 + 1 = buffer_count - 2
	for(int i = R_EDGE - 2; i >= 0; i--)
	{
		if(threadId == 0)
			divider = load_r_mat(r_mat, i, i);
		
		SyncThreads();
		
		#if COMPRESSED_R
		if(threadId < R_EDGE && threadId >= i)
		#else
		// First values are always zero if R !COMPRESSED_R and
		// "&& threadId >= i" makes not compressed code run little bit slower
		if(threadId < R_EDGE)
		#endif
		{
			vec3 value = load_r_mat(r_mat, threadId, i);
			store_r_mat(r_mat, threadId, i, value / vec3(divider.x, divider.y, divider.z));
		}

		SyncThreads();

		#if 1 // ORIGINAL
		if(threadId == 0) // Optimization proposal: parallel reduction
		{
			for(int j = i + 1; j < R_EDGE - 1; j++)
			{
				vec3 value  = load_r_mat(r_mat, R_EDGE - 1, i);
				vec3 value2 = load_r_mat(r_mat, j, i);
				store_r_mat(r_mat, R_EDGE - 1, i, value - value2);
			}
		}
		#else
		const int startRIdx = (i + 1);
		const int endRIdx	= (R_EDGE - 1);
		const int numItems	= endRIdx - startRIdx;
		if(threadId < numItems)
		{
			// Parallel load
			const int j = startRIdx + threadId;
			vec3 value2 = load_r_mat(r_mat, j, i);

			// Then iterate over active threads and gather data from lane 0
			if(threadId == 0)
			{
				for(int k = startRIdx; k < endRIdx; k++)
				{
					vec3 value = load_r_mat(r_mat, R_EDGE - 1, i);
					vec3 currValue2 = ...;// load from lane k
					store_r_mat(r_mat, R_EDGE - 1, i, value - currValue2);
				}
			}
		}
		#endif

		SyncThreads();

		#if COMPRESSED_R
		if(threadId < R_EDGE && i >= threadId)
		#else
		if(threadId < R_EDGE)
		#endif
		{
			vec3 value  = load_r_mat(r_mat, i, threadId);
			vec3 value2 = load_r_mat(r_mat, R_EDGE - 1, i);
			store_r_mat(r_mat, i, threadId, value * value2);
		}
		SyncThreads();
	}

	// The features are stored in the first (buffers-3) values: the last 3 contain the noisy 1spp color channels
	if(threadId < BUFFER_COUNT - 3)
	{
		// Store weights
		const int index = groupId * (BUFFER_COUNT - 3) + threadId;
		const vec3 weight = load_r_mat(r_mat, R_EDGE - 1, threadId);
		store_float3(weights, index, weight);
	}
}

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
)
{
	fitter<<<grid_size, block_size>>>(
		params,
		weights,
		mins_maxs,
		features_buffer
	);
}

// Weighted sum kernel /////////////////////////////////////////////////////////
// -> outputs the noise-free 1spp color estimate

__global__ void weighted_sum(
	WeightedSumKernelParams params,
	const float * K_RESTRICT weights,			// [in]	 Features weights computed by the fitter kernel
	const float * K_RESTRICT mins_maxs,			// [in]  Min and max of features values per block (world_positions)
		  float * K_RESTRICT output,			// [out] Noise-free color estimate
	const float * K_RESTRICT current_normals,	// [in]  Current (world) normals
	const float * K_RESTRICT current_positions	// [in]  Current world positions
)
{
	const ivec2 pixel = ivec2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	
	const int w = params.sizeX;
	const int h = params.sizeY;

	if(pixel.x >= w || pixel.y >= h)
		return;

	// Linear pixel index
	const int linear_pixel = pixel.y * w + pixel.x;

	// Retrieve linear group index from the offset pixel
	const ivec2 offset_pixel = pixel + BLOCK_EDGE_HALF - BLOCK_OFFSETS[params.frameNumber % BLOCK_OFFSETS_COUNT];
	const int group_index = (offset_pixel.x / BLOCK_EDGE_LENGTH) + (offset_pixel.y / BLOCK_EDGE_LENGTH) * params.worksetWithMarginBlockCountX;

	// Reload features from buffer here to have values without stochastic regularization noise
	// TODO: bind the normalized world_position buffer to avoid renormalizing again (no need for mins_maxs buffer)
	vec3 world_position = load_float3(current_positions, linear_pixel); 
	vec3 normal = load_float3(current_normals, linear_pixel);
	float features[BUFFER_COUNT - 3] =
	{
		// TODO: replace with function that fill the array?
		FEATURE_BUFFERS // expands to 1.f, normal.x, ..., world_position.x, ..., world_position.x * world_position.x, ...
	};

	// Weighted sum of the feature buffers
	vec3 color = vec3(0.f, 0.f, 0.f);
	for(int feature_buffer = 0; feature_buffer < BUFFER_COUNT - 3; feature_buffer++)
	{
		float feature = features[feature_buffer];

		// Scale world position buffers
		if(feature_buffer >= FEATURES_NOT_SCALED)
		{
			const int min_max_index = (group_index * FEATURES_SCALED + feature_buffer - FEATURES_NOT_SCALED) * 2;
			feature = scale(feature, mins_maxs[min_max_index + 0], mins_maxs[min_max_index + 1]);
		}

		// Load weight and sum
		vec3 weight = load_float3(weights, group_index * (BUFFER_COUNT - 3) + feature_buffer);
		color += weight * feature;
	}

	// Remove negative values from every component of the fitting results
	color = Max(vec3(0.f), color);

	// Store results
	store_float3(output, linear_pixel, color);
}


extern "C" void run_weighted_sum(
	dim3 const & grid_size,
	dim3 const & block_size,
	WeightedSumKernelParams const & params,
	const float * K_RESTRICT weights,			// [in]	 Features weights computed by the fitter kernel
	const float * K_RESTRICT mins_maxs,			// [in]  Min and max of features values per block (world_positions)
		  float * K_RESTRICT output,			// [out] Noise-free color estimate
	const float * K_RESTRICT current_normals,	// [in]  Current (world) normals
	const float * K_RESTRICT current_positions	// [in]  Current world positions
)
{
	weighted_sum<<<grid_size, block_size>>>(
		params,
		weights,
		mins_maxs,
		output,
		current_normals,
		current_positions
	);
}

// Accumulate filtered data kernel /////////////////////////////////////////////
// -> outputs the noise-free accumulated color estimate + a tonemapped version w/ albedo

// TODO: make 2 versions: one for frame 0 and one for the rest (avoid a branch)

__global__ void accumulate_filtered_data(
	AccumulateFilteredDataKernelParams params,
	const float * K_RESTRICT filtered_frame,			// [in]  Noise free color estimate (computed as the weighted sum of the features)
	const vec2 * K_RESTRICT in_prev_frame_pixel,		// [in]  Previous frame pixel coordinates (after reprojection)
	const unsigned char * K_RESTRICT accept_bools,		// [in]  Validity mask of bilinear samples in previous frame (after reprojection)
	const float * K_RESTRICT albedo_buffer,				// [in]  Albedo buffer of the current frame (non-noisy)
		  float * K_RESTRICT tone_mapped_frame,			// [out] Accumulated and tonemapped noise-free color estimate
	const unsigned char* K_RESTRICT current_spp,		// [in]	 Current number of samples accumulated (for CMA)
	const float * K_RESTRICT accumulated_prev_frame,	// [in]  Previous frame noise-free accumulated color estimate 
		  float * K_RESTRICT accumulated_frame			// [out] Current frame noise-free accumulated color estimate
)
{
	const ivec2 pixel = ivec2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	
	const int w = params.sizeX;
	const int h = params.sizeY;
	
	if(pixel.x >= w || pixel.y >= h)
		return;

	// Linear pixel index
	const unsigned int linear_pixel = pixel.y * w + pixel.x;

	// Noise-free estimate of the color (computed via a weighted sum of features)
	vec3 filtered_color = load_float3(filtered_frame, linear_pixel);
	vec3 prev_color = vec3(0.f, 0.f, 0.f);
	float blend_alpha = 1.f;

	// Reproject and accumulate previous frame noise-free estimate
	if(params.frameNumber > 0)
	{
		// Bitmask telling which bilinear samples were accepted in the first accumulation kernel
		const unsigned char accept = accept_bools[linear_pixel];

		if(accept > 0) // If any prev frame sample is accepted
		{
			// Pixel coordinates in the previous frame (in [0, w-1]x[0, h-1])
			const vec2 prev_frame_pixel_f = in_prev_frame_pixel[linear_pixel];
			
			// Integer pixel coordinates in the previous frame
			const ivec2 prev_frame_pixel_i = FloatToIntRd(prev_frame_pixel_f);

			// Compute bilinear weights for bilinear sampling
			const vec2 prev_pixel_fract = prev_frame_pixel_f - vec2(prev_frame_pixel_i);
			const vec2 one_minus_prev_pixel_fract = 1.f - prev_pixel_fract;

			float total_weight = 0.f;

			// Add valid bilinear samples

			if(accept & 0x01)
			{
				float weight = one_minus_prev_pixel_fract.x * one_minus_prev_pixel_fract.y;
				int linear_sample_location = prev_frame_pixel_i.y * w + prev_frame_pixel_i.x;
				prev_color += weight * load_float3(accumulated_prev_frame, linear_sample_location);
				total_weight += weight;
			}

			if(accept & 0x02)
			{
				float weight = prev_pixel_fract.x * one_minus_prev_pixel_fract.y;
				int linear_sample_location = prev_frame_pixel_i.y * w + prev_frame_pixel_i.x + 1;
				prev_color += weight * load_float3(accumulated_prev_frame, linear_sample_location);
				total_weight += weight;
			}

			if(accept & 0x04)
			{
				float weight = one_minus_prev_pixel_fract.x * prev_pixel_fract.y;
				int linear_sample_location = (prev_frame_pixel_i.y + 1) * w + prev_frame_pixel_i.x;
				prev_color += weight * load_float3(accumulated_prev_frame, linear_sample_location);
				total_weight += weight;
			}

			if(accept & 0x08)
			{
				float weight = prev_pixel_fract.x * prev_pixel_fract.y;
				int linear_sample_location = (prev_frame_pixel_i.y + 1) * w + prev_frame_pixel_i.x + 1;
				prev_color += weight * load_float3(accumulated_prev_frame, linear_sample_location);
				total_weight += weight;
			}

			if(total_weight > 0.f)
			{
				// Blend_alpha is dymically decided so that the result is average
				// of all samples (cumulative moving average) until the cap defined by
				// SECOND_BLEND_ALPHA is reached (exponential moving average: EMA_(n+1) = (1 - a) * EMA_n + a * x_(n+1) = lerp(EMA_n, x_(n+1), a))

				// [Section 3.5]
				// Similarly to the first temporal accumulation we use the cumulative moving average until
				// the weight of the new sample has reached the chosen SECOND_BLEND_ALPHA (e.g 10%).
				// Using the cumulative moving average in this second temporal accumulation is crucial since
				// the first block fitted after an occlusion is more likely to contain outlier data and with
				// the cumulative moving average it is mixed with subsequent frames more quickly.
				blend_alpha = 1.f / float(current_spp[linear_pixel]);
				blend_alpha = Max(blend_alpha, SECOND_BLEND_ALPHA);
				prev_color /= total_weight;

				// Note: we accumulate at most 255 samples for the cumulative moving average (which is more than enough because of
				// the threshold SECOND_BLEND_ALPHA that switch to exponential moving average).
				// E.g: SECOND_BLEND_ALPHA = 0.1 = 1 / (n + 1) <=> n = (1 - 0.1) / 0.1 = 9 => above 9 samples for a pixel,
				// we switch to exponential moving average with alpha = 10%
			}
		}
	}

	// Mix with colors and store results
	vec3 accumulated_color = blend_alpha * filtered_color + (1.f - blend_alpha) * prev_color; // Lerp(prev_color, filtered_color, blend_alpha);
	store_float3(accumulated_frame, linear_pixel, accumulated_color);

	// Remodulate albedo and tone map
	vec3 albedo = load_float3(albedo_buffer, linear_pixel);
	const vec3 tone_mapped_color = Clamp(Pow(Max(vec3(0.f), albedo * accumulated_color), 0.454545f), vec3(0.f), vec3(1.f));
	store_float3(tone_mapped_frame, linear_pixel, tone_mapped_color);
}

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
)
{
	accumulate_filtered_data<<<grid_size, block_size>>>(
		params,
		filtered_frame,
		in_prev_frame_pixel,
		accept_bools,
		albedo_buffer,
		tone_mapped_frame,
		current_spp,
		accumulated_prev_frame,
		accumulated_frame
	);
}

// TAA kernel //////////////////////////////////////////////////////////////////

// TODO:
// - make two versions of the kernel: one for the frame 0 and one for the rest
// - optimize with local/shared memory
__global__ void taa(
	TAAKernelParams params,
	const vec2 * K_RESTRICT in_prev_frame_pixel,	// [in]  Previous frame pixel coordinates (after reprojection)
	const float * K_RESTRICT new_frame,				// [in]	 Current frame color buffer
		  float * K_RESTRICT result_frame,			// [out] Antialiased frame color buffer
	const float * K_RESTRICT prev_frame				// [in]  Previous frame color buffer
)
{
	const ivec2 pixel = ivec2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
   
	const int w = params.sizeX;
	const int h = params.sizeY;

	if(pixel.x >= w || pixel.y >= h)
		return;

	// Linear pixel index
	const unsigned int linear_pixel = pixel.y * w + pixel.x;

	// Current frame color
	vec3 my_new_color = load_float3(new_frame, linear_pixel);

	// Previous frame pixel coordinates
	const vec2 prev_frame_pixel_f = in_prev_frame_pixel[linear_pixel];
	const ivec2 prev_frame_pixel_i = FloatToIntRd(prev_frame_pixel_f);

	// Return if all sampled pixels are going to be out of image area
	if(params.frameNumber == 0 ||
		prev_frame_pixel_i.x < -1 || prev_frame_pixel_i.y < -1 ||
		prev_frame_pixel_i.x >= w || prev_frame_pixel_i.y >= h
	)
	{
		store_float3(result_frame, linear_pixel, my_new_color);
		return;
	}

	// Compute the color AABB in the 3x3 neighbourhood and the min/max in a cross pattern around the current pixel
	vec3 minimum_box	= vec3(+C_FLT_MAX);
	vec3 minimum_cross	= vec3(+C_FLT_MAX);
	vec3 maximum_box	= vec3(-C_FLT_MAX);
	vec3 maximum_cross	= vec3(-C_FLT_MAX);
	for(int y = -1; y <= 1; ++y)
	{
		for(int x = -1; x <= 1; ++x)
		{
			ivec2 sample_location = pixel + ivec2(x, y);
			if(sample_location.x >= 0 && sample_location.y >= 0 &&
			   sample_location.x < w && sample_location.y < h
			)
			{
				vec3 sample_color;
				if(x == 0 && y == 0)
					sample_color = my_new_color;
				else
					sample_color = load_float3(new_frame, sample_location.x + sample_location.y * w);

				sample_color = RGB_to_YCoCg(sample_color);

				if(x == 0 || y == 0)
				{
					minimum_cross = Min(minimum_cross, sample_color);
					maximum_cross = Max(maximum_cross, sample_color);
				}

				minimum_box = Min(minimum_box, sample_color);
				maximum_box = Max(maximum_box, sample_color);
			}
		}
	}

	// Bilinear sampling of previous frame.
	// Note: work-item has already returned if the sampling location is completly out of image
	vec3 prev_color = vec3(0.f, 0.f, 0.f);
	float total_weight = 0;
	const vec2 pixel_fract = prev_frame_pixel_f - vec2(prev_frame_pixel_i);
	const vec2 one_minus_pixel_fract = 1.f - pixel_fract;

	if(prev_frame_pixel_i.y >= 0)
	{
		if(prev_frame_pixel_i.x >= 0)
		{
			float weight = one_minus_pixel_fract.x * one_minus_pixel_fract.y;
			prev_color += weight * load_float3(prev_frame, prev_frame_pixel_i.y * w + prev_frame_pixel_i.x);
			total_weight += weight;
		}

		if(prev_frame_pixel_i.x < w - 1)
		{
			float weight = pixel_fract.x * one_minus_pixel_fract.y;
			prev_color += weight * load_float3(prev_frame, prev_frame_pixel_i.y * w + prev_frame_pixel_i.x + 1);
			total_weight += weight;
		}
	}

	if(prev_frame_pixel_i.y < h - 1)
	{
		if(prev_frame_pixel_i.x >= 0)
		{
			float weight = one_minus_pixel_fract.x * pixel_fract.y;
			prev_color += weight * load_float3(prev_frame, (prev_frame_pixel_i.y + 1) * w + prev_frame_pixel_i.x);
			total_weight += weight;
		}

		if(prev_frame_pixel_i.x < w - 1)
		{
			float weight = pixel_fract.x * pixel_fract.y;
			prev_color += weight * load_float3(prev_frame, (prev_frame_pixel_i.y + 1) * w + prev_frame_pixel_i.x + 1);
			total_weight += weight;
		}
	}

	if(total_weight > 0)
		prev_color /= total_weight; // Total weight can be less than one on the edges

	vec3 prev_color_ycocg = RGB_to_YCoCg(prev_color);

	// Note: Some references use more complicated methods to move the previous frame color to the YCoCg space AABB
	vec3 minimum = (minimum_box + minimum_cross) / 2.f;
	vec3 maximum = (maximum_box + maximum_cross) / 2.f;
	vec3 prev_color_rgb = YCoCg_to_RGB(Clamp(prev_color_ycocg, minimum, maximum));

	vec3 result_color = TAA_BLEND_ALPHA * my_new_color + (1.f - TAA_BLEND_ALPHA) * prev_color_rgb; // Lerp(prev_color_rgb, my_new_color, TAA_BLEND_ALPHA);
	store_float3(result_frame, linear_pixel, result_color);
}

extern "C" void run_taa(
	dim3 const & grid_size,
	dim3 const & block_size,
	TAAKernelParams const & params,
	const vec2 * K_RESTRICT in_prev_frame_pixel,	// [in]  Previous frame pixel coordinates (after reprojection)
	const float * K_RESTRICT new_frame,				// [in]	 Current frame color buffer
		  float * K_RESTRICT result_frame,			// [out] Antialiased frame color buffer
	const float * K_RESTRICT prev_frame				// [in]  Previous frame color buffer
)
{
	taa<<<grid_size, block_size>>>(
		params,
		in_prev_frame_pixel,
		new_frame,
		result_frame,
		prev_frame
	);
}
