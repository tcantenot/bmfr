

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

inline __device__ void SyncThreads()
{
	__syncthreads();
}

// Parallel reductions /////////////////////////////////////////////////////////

// Unrolled parallel sum reduction of 256 values
// TODO: unused start_index...
inline __device__ void parallel_reduction_sum_256(float * result, volatile float * pr_data_256, const int start_index)
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
inline __device__ void parallel_reduction_min_256(float * result, volatile float * pr_data_256)
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
inline __device__ void parallel_reduction_max_256(float * result, volatile float * pr_data_256)
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
static const ivec2 BLOCK_OFFSETS[BLOCK_OFFSETS_COUNT] = {
	ivec2( -14, -14),
	ivec2(   4,  -6),
	ivec2(  -8,  14),
	ivec2(   8,   0),
	ivec2( -10,  -8),
	ivec2(   2,  12),
	ivec2(  12, -12),
	ivec2( -10,   0),
	ivec2(  12,  14),
	ivec2(  -8, -16),
	ivec2(   6,   6),
	ivec2(  -2,  -2),
	ivec2(   6, -14),
	ivec2( -16,  12),
	ivec2(  14,  -4),
	ivec2(  -6,   4)
};


// Features buffer indexing ////////////////////////////////////////////////////

// TODO: change these defines either by macro that take parameters or inline functions
// Helper defines ONLY used in IN_ACCESS define
#define HORIZONTAL_BLOCKS (WORKSET_WIDTH / BLOCK_EDGE_LENGTH)
#define BLOCK_INDEX_X (group_id % (HORIZONTAL_BLOCKS + 1))
#define BLOCK_INDEX_Y (group_id / (HORIZONTAL_BLOCKS + 1))
#define IN_BLOCK_INDEX (BLOCK_INDEX_Y * (HORIZONTAL_BLOCKS + 1) + BLOCK_INDEX_X)
#define FEATURE_START (feature_buffer * BLOCK_PIXELS)

#define IN_ACCESS (IN_BLOCK_INDEX * buffers * BLOCK_PIXELS + FEATURE_START + sub_vector * 256 + id)


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

inline __device__ vec3 load_r_mat(const vec3 * r_mat, const int x, const int y)
{
   return r_mat[R_ACCESS];
}

inline __device__ void store_r_mat(volatile vec3* r_mat, const int x, const int y, const vec3 value)
{
   r_mat[R_ACCESS] = value;
}

inline __device__ void store_r_mat_broadcast(volatile vec3 * r_mat, const int x, const int y, const float value)
{
   r_mat[R_ACCESS] = value;
}

inline __device__ void store_r_mat_channel(volatile vec3 * r_mat, const int x, const int y, const int channel, const float value)
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
inline __device__ float random(unsigned int a)
{
	a = (a+0x7ed55d16) + (a<<12);
	a = (a^0xc761c23c) ^ (a>>19);
	a = (a+0x165667b1) + (a<<5);
	a = (a+0xd3a2646c) ^ (a<<9);
	a = (a+0xfd7046c5) + (a<<3);
	a = (a^0xb55a4f09) ^ (a>>16);

	return convert_float(a) / convert_float(UINT_MAX);
}

inline __device__ float add_random(
	const float value,
	const int id,
	const int sub_vector,
	const int feature_buffer,
	const int frame_number
)
{
	float seed = id + sub_vector * LOCAL_SIZE + feature_buffer * BLOCK_EDGE_LENGTH * BLOCK_EDGE_LENGTH +
				 frame_number * BUFFER_COUNT * BLOCK_EDGE_LENGTH * BLOCK_EDGE_LENGTH;
	float noise01 = random(seed);
	float signedZeroMeanNoise = 2.f * noise01 - 1.f;
	return value + NOISE_AMOUNT * signedZeroMeanNoise;
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

// Alternative scale method
inline __device__ float Scale01(float value, float min, float max)
{
	const float span = max - min;
	return (span > 0.0f) ? (value - min) / span : 0.0f;
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

inline __device__ half2 FloatToHalf(vec2 v)
{
	return half2(__float2half(v.x), __float2half(v.y));
}

inline __device__ int FloatToIntRn(float x)
{
	return __float2int_rn(x);
}

inline __device__ ivec2 FloatToIntRn(vec2 v)
{
	return ivec2(__float2int_rn(v.x), __float2int_rn(v.y));
}

inline __device__ ivec3 FloatToIntRn(vec3 v)
{
	return ivec3(__float2int_rn(v.x), __float2int_rn(v.y), __float2int_rn(v.z));
}

inline __device__ ivec4 FloatToIntRn(vec4 v)
{
	return ivec4(__float2int_rn(v.x), __float2int_rn(v.y), __float2int_rn(v.z), __float2int_rn(v.w));
}

// Load/store vec3 functions /////////////////////////////////////////////////

inline void store_float3(volatile float * __restrict__ buffer, const int index, const vec3 value)
{
	buffer[index * 3 + 0] = value.x;
	buffer[index * 3 + 1] = value.y;
	buffer[index * 3 + 2] = value.z;
}

// This is significantly slower the the inline function on Vega FE
//#define store_float3(buffer, index, value) \
//   buffer[(index) * 3 + 0] = value.x; \
//   buffer[(index) * 3 + 1] = value.y; \
//   buffer[(index) * 3 + 2] = value.z

#define load_float3(buffer, index) vec3(buffer[(index) * 3], buffer[(index) * 3 + 1], buffer[(index) * 3 + 2])

// inline __device__ vec3 load_float3(volatile float * __restrict__ buffer, const int index)
// {
//   return vec3(buffer[index * 3 + 0], buffer[index * 3 + 1], buffer[index * 3 + 2]);
// }

#if USE_HALF_PRECISION_IN_FEATURES_DATA
#define LOAD(buffer, index) buffer[index]
#define STORE(buffer, index, value) buffer[index] = FloatToHalf(value)
#else
#define LOAD(buffer, index) buffer[index]
#define STORE(buffer, index, value) buffer[index] = value
#endif


// Accumulate noisy 1spp color kernel //////////////////////////////////////////

__global__ void accumulate_noisy_data(
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
)
{
	const ivec2 gid = ivec2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

	// Mirror indexed of the input. x and y are always less than one size out of
	// bounds if image dimensions are bigger than BLOCK_EDGE_LENGTH
	// BLOCK_EDGE_HALF = half block size (32/2 -> 16)
	const ivec2 pixel_without_mirror = gid - BLOCK_EDGE_HALF + BLOCK_OFFSETS[frame_number % BLOCK_OFFSETS_COUNT]; // TODO: input directly frame_number % BLOCK_OFFSETS_COUNT

	// Pixel coordinates in [0, IMAGE_WIDTH-1]x[0, IMAGE_HEIGHT-1]
	const ivec2 pixel = mirror2(pixel_without_mirror, ivec2(IMAGE_WIDTH, IMAGE_HEIGHT));

	// Linear pixel index in image in [0, IMAGE_WIDTH*IMAGE_HEIGHT-1]
	const int linear_pixel = pixel.y * IMAGE_WIDTH + pixel.x;
   
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
	vec3 current_color = load_float3(current_noisy, linear_pixel);

	// Current frame world position
	vec4 world_position = vec4(load_float3(current_positions, linear_pixel), 1.f);

	// Current frame (world) normal
	vec3 normal = load_float3(current_normals, linear_pixel);


	// Previous frame pixel coordinates in [0, IMAGE_WIDTH-1]x[0, IMAGE_HEIGHT-1]
	// Default is the same pixel coordinates (no movement)
	vec2 prev_frame_pixel_f = vec2(pixel.x, pixel.y);

	// Bit mask telling which previous frame (bilinear) samples are valid under reprojection into current frame
	unsigned char store_accept = 0x00;

	// Blending factor with history buffer
	// Blend_alpha 1.f means that only current frame color is used. The value is changed if sample from previous frame can be used
	float blend_alpha = 1.f;
	vec3 previous_color = vec3(0.f, 0.f, 0.f);

	float sample_spp = 0.f;
	if(frame_number > 0)
	{
		// Project current world position into previous frame with the previous ViewProj matrix

		// TODO: check if the computation is correct + transpose on CPU to avoid contructing a vec4 for each row
		// Matrix multiplication and normalization to 0..1
		vec2 prev_frame_uv;
		prev_frame_uv.x = Dot(prev_frame_camera_matrix.row(0), world_position); // Transform x
		prev_frame_uv.y = Dot(prev_frame_camera_matrix.row(1), world_position); // Transform y
		// No need for z-buffer in accumulation of the noisy data
		// -> might be useful if we use it to detect disocclusion
		// --> compare previous z (store previous frame Z-buffer) with prev_frame_pixel_uv.z
		//prev_frame_uv.z = Dot(prev_frame_camera_matrix.row(2), world_position); // Transform z
		prev_frame_uv /= Dot(prev_frame_camera_matrix.row(3), world_position);
		prev_frame_uv += 1.f; prev_frame_uv /= 2.f; // prev_frame_uv = prev_frame_uv * 0.5f + 0.5f;

		// Compute the pixel coordinates in the previous frame (in [0, IMAGE_WIDTH-1]x[0, IMAGE_HEIGHT-1])
		prev_frame_pixel_f = prev_frame_uv * vec2(IMAGE_WIDTH, IMAGE_HEIGHT);

		// Apply offset (TODO: what offset??? seems to always be 0.5... Maybe TAA/ray subpixel offsets -> send 1-pixel_offset.y)
		// TODO: try to remove this
		prev_frame_pixel_f -= vec2(pixel_offset.x, 1 - pixel_offset.y);

		// Convert into integer pixel coordinates (round to nearest)
		ivec2 prev_frame_pixel_i = FloatToIntRn(prev_frame_pixel_f);

		// Compute bilinear weights (for bilinear sampling)
		// TODO: implement bicubic Catmull-Rom (for sharpness)? => would need to perform more fetches and store more "validity bits" in mask
		const ivec2 offsets[4] = { ivec2(0, 0), ivec2(1, 0), ivec2(0, 1), ivec2(1, 1) };

		vec2 prev_pixel_fract = prev_frame_pixel_f - vec2(prev_frame_pixel_i);
		vec2 one_minus_prev_pixel_fract = 1.f - prev_pixel_fract;
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
			int linear_sample_location = sample_location.y * IMAGE_WIDTH + sample_location.x;

			// Check if previous frame color can be used based on its screen location
			if(sample_location.x >= 0 && sample_location.y >= 0 &&
			   sample_location.x < IMAGE_WIDTH && sample_location.y < IMAGE_HEIGHT
			)
			{
				// Fetch previous frame world position
				vec3 prev_world_position = load_float3(previous_positions, linear_sample_location);

				// Compute world distance squared
				vec3 position_difference = prev_world_position - world_position.xyz;
				float position_distance_squared = Dot(position_difference, position_difference);

				// World position distance discard
				if(position_distance_squared < float(POSITION_LIMIT_SQUARED)){

					// Fetch previous frame normal
					vec3 prev_normal = load_float3(previous_normals, linear_sample_location);

					// Distance of the normals
					// NOTE: could use some other distance metric (e.g. angle), but we use hard
					// experimentally found threshold -> means that the metric doesn't matter.
					vec3 normal_difference = prev_normal - normal;
					float normal_distance_squared = Dot(normal_difference, normal_difference);

					if(normal_distance_squared < float(NORMAL_LIMIT_SQUARED)){

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
		new_spp = (sample_spp > 254.f) ? 255 : static_cast<unsigned char>(sample_spp) + 1;
	}
	current_spp[linear_pixel] = new_spp; // Store current number of samples accumulated (for CMA)

	vec3 new_color = blend_alpha * current_color + (1.f - blend_alpha) * previous_color; // Lerp(previous_color, current_color, blend_alpha);


	// The set of feature buffers used in the fitting
	float features[BUFFER_COUNT] =
	{
		FEATURE_BUFFERS, // expands to 1.f, normal.x, ..., world_position.x, ..., world_position.x * world_position.x, ...
		new_color.x,
		new_color.y,
		new_color.z
	};

	for(int feature_num = 0; feature_num < BUFFER_COUNT; ++feature_num)
	{
		const int x_block = gid.x / BLOCK_EDGE_LENGTH; // Block coordinate x
		const int y_block = gid.y / BLOCK_EDGE_LENGTH; // Block coordinate y
		const int x_in_block = gid.x % BLOCK_EDGE_LENGTH; // Thread coordinate x inside block in [0, BLOCK_EDGE_LENGTH-1]
		const int y_in_block = gid.y % BLOCK_EDGE_LENGTH; // Thread coordinate y inside block in [0, BLOCK_EDGE_LENGTH-1]

		// Index in feature buffer (data are concatenated)
		const unsigned int location_in_data = feature_num * BLOCK_PIXELS + 
			x_in_block + y_in_block * BLOCK_EDGE_LENGTH +
			x_block * BLOCK_PIXELS * BUFFER_COUNT +
			y_block * (WORKSET_WITH_MARGINS_WIDTH / BLOCK_EDGE_LENGTH) *
			BLOCK_PIXELS * BUFFER_COUNT;

		#if 0
		// | Block 0 feature 0 | Block 0 feature 0 | ... | Block 0 feature M | ... | Block 1 feature 0 | ... | Block N feature 0 | ... | Block N feature M |
		const unsigned int numBlockX = WORKSET_WITH_MARGINS_WIDTH / BLOCK_EDGE_LENGTH;
		const unsigned int location_in_data = y_block * (WORKSET_WITH_MARGINS_WIDTH / BLOCK_EDGE_LENGTH) * BLOCK_PIXELS * BUFFER_COUNT +
											  x_block * BLOCK_PIXELS * BUFFER_COUNT +
											  feature_num * BLOCK_PIXELS +
											  y_in_block * BLOCK_EDGE_LENGTH +
											  x_in_block;
		#endif

		float feature = features[feature_num];

		if(isnan(feature))
			feature = 0.0f;

		#if USE_HALF_PRECISION_IN_FEATURES_DATA
		feature = Clamp(feature, -65504.f, 65504.f);
		#endif

		STORE(features_data, location_in_data, feature);
	}

	// The kernel works on a workset of size WORKSET_WITH_MARGINS_WIDTH x WORKSET_WITH_MARGINS_HEIGHT
	// -> the extra block margin is used to handle the offsets applied to reduce the block artifacts.
	// [Section 3.5]: "To aid the reduction of blockiness, BMFR processes each frame over a grid of non-overlapping
	//	blocks which is displaced with random offsets. These offsets prevent the artifacts that would arise from reusing
	// same block positions on a static scene with a static camera."
	// --> Only the pixels inside the image (after applying the offsets) should write to the output data that
	// have the same size of the input image
	if(pixel_without_mirror.x >= 0 && pixel_without_mirror.x < IMAGE_WIDTH &&
	   pixel_without_mirror.y >= 0 && pixel_without_mirror.y < IMAGE_HEIGHT
	)
	{
		store_float3(current_noisy, linear_pixel, new_color); // Accumulated noisy 1spp
		out_prev_frame_pixel[linear_pixel] = prev_frame_pixel_f; // Previous frame pixel coordinates (to sample history)
		accept_bools[linear_pixel] = store_accept; // "History validity" bitmask
	}
}
