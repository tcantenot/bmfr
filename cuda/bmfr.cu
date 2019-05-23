

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
// and variables (values that depend on RT size)

// Threads synchronization /////////////////////////////////////////////////////

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
	float * __restrict__ weights,					// [out] Features weights
	float * __restrict__ mins_maxs,					// [out] Min and max of features values per block (world_positions)
	#if USE_HALF_PRECISION_IN_FEATURES_DATA
	half * __restrict__ features_buffer,			// [out] Features buffer (half-precision)
	#else
	float * __restrict__ features_buffer,			// [out] Features buffer (single-precision)
	#endif
	const int frame_number							// [in]  Current frame number
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
	__shared__ vec3 r_mat_sdata[R_SHARED_DATA_SIZE];	// Shared memory used to store the R matrices of the QR factorization (vec3 -> one per color channel)
	__shared__ float u_length_squared;					// Shared memory variable that holds the 'u' vector square length
	__shared__ float dotProd;							// Shared memory variable that holds the dot product of...
	__shared__ float block_min;							// Shared memory variable that holds the result of the parallel min reduction
	__shared__ float block_max;							// Shared memory variable that holds the result of the parallel max reduction
	__shared__ float vec_length;						// Shared memory variable that holds the vec length			

	float * pr_data_256 = &pr_shared_data[0];
	float * u_vec = &u_vec_sdata[0];
	vec3 * r_mat = &r_mat_sdata[0];

	const int group_id = blockIdx.x;
	const int id = threadIdx.x; // in [0, 255]
	const int buffers = BUFFER_COUNT;

	// Scales world positions to 0..1 in a block
	const int feature_to_scale_beg_idx = FEATURES_NOT_SCALED;
	const int feature_to_scale_end_idx = buffers - 3;
	for(int feature_buffer = feature_to_scale_beg_idx; feature_buffer < feature_to_scale_end_idx; ++feature_buffer)
	{
		// Find maximum and minimum of the whole block
		float tmp_max = -C_FLT_MAX;
		float tmp_min = +C_FLT_MAX;
	  
		// LOCAL_SIZE = size of the shared memory (= 256)
		// BLOCK_PIXELS = number of pixels in a block (32x32 = 1024)

		// #define HORIZONTAL_BLOCKS (WORKSET_WIDTH / BLOCK_EDGE_LENGTH)
		// #define BLOCK_INDEX_X (group_id % (HORIZONTAL_BLOCKS + 1))
		// #define BLOCK_INDEX_Y (group_id / (HORIZONTAL_BLOCKS + 1))
		// #define IN_BLOCK_INDEX (BLOCK_INDEX_Y * (HORIZONTAL_BLOCKS + 1) + BLOCK_INDEX_X)
		// #define FEATURE_START (feature_buffer * BLOCK_PIXELS)
		// #define IN_ACCESS (IN_BLOCK_INDEX * buffers * BLOCK_PIXELS + FEATURE_START + sub_vector * LOCAL_SIZE + id)

		// Manual unrolling for parallel reduction as the block contains 1024 (32x32) work items and
		// the reduction operates on 256 elements (group size)
		// -> Compute the min and max of N values (N = 1024/256 = 4)
		const int N = BLOCK_PIXELS / LOCAL_SIZE;
		for(int sub_vector = 0; sub_vector < N; ++sub_vector)
		{
			float value = LOAD(features_buffer, IN_ACCESS);
			tmp_max = Max(value, tmp_max);
			tmp_min = Min(value, tmp_min);
		}

		// Parallel min reduction
		pr_data_256[id] = tmp_min;
		SyncThreads();
		parallel_reduction_min_256(&block_min, pr_data_256);

		// Parallel max reduction
		pr_data_256[id] = tmp_max;
		SyncThreads();
		parallel_reduction_max_256(&block_max, pr_data_256);

		// Output the min and max features values per block of 32x32 pixels (only output 256 values because of manual unrolling of 4)
		if(id == 0)
		{
			const int index = (group_id * FEATURES_SCALED + (feature_buffer - feature_to_scale_beg_idx)) * 2;
			mins_maxs[index + 0] = block_min;
			mins_maxs[index + 1] = block_max;
		}
		SyncThreads(); // TODO: this thread synchronization may not be useful

		// Scale feature and replace value in features buffer
		for(int sub_vector = 0; sub_vector < BLOCK_PIXELS / LOCAL_SIZE; ++sub_vector)
		{
			float scaled_value = scale(LOAD(features_buffer, IN_ACCESS), block_min, block_max);
			STORE(features_buffer, IN_ACCESS, scaled_value);
		}
	}

	// TODO: move this decision to the CPU + set define
	// Non square matrices require processing every column. Otherwise result is
	// OKish, but R is not upper triangular matrix
	int limit = buffers == BLOCK_PIXELS ? buffers - 1 : buffers;

	// Compute R
	for(int col = 0; col < limit; col++)
	{
		// Note: the last 3 features values are the 3 channels of the color (not used for the regression)
		int col_limited = Min(col, buffers - 3);

		// Load new column into memory
		int feature_buffer = col;
		float tmp_sum_value = 0.f;

		// LOCAL_SIZE = size of the shared memory (= 256)
		// BLOCK_PIXELS = number of pixels in a block (32x32 = 1024)

		// #define HORIZONTAL_BLOCKS (WORKSET_WIDTH / BLOCK_EDGE_LENGTH)
		// #define BLOCK_INDEX_X (group_id % (HORIZONTAL_BLOCKS + 1))
		// #define BLOCK_INDEX_Y (group_id / (HORIZONTAL_BLOCKS + 1))
		// #define IN_BLOCK_INDEX (BLOCK_INDEX_Y * (HORIZONTAL_BLOCKS + 1) + BLOCK_INDEX_X)
		// #define FEATURE_START (feature_buffer * BLOCK_PIXELS)
		// #define IN_ACCESS (IN_BLOCK_INDEX * buffers * BLOCK_PIXELS + FEATURE_START + sub_vector * LOCAL_SIZE + id)

		// Manual unrolling for parallel reduction as the block contains 1024 (32x32) work items and
		// the reduction operates on 256 elements (group size)
		// -> Compute the sum of N values (N = 1024/256 = 4)
		for(int sub_vector = 0; sub_vector < BLOCK_PIXELS / LOCAL_SIZE; ++sub_vector)
		{
			// Load feature
			float tmp = LOAD(features_buffer, IN_ACCESS);

			// Store the feature in shared memory
			const int index = id + sub_vector * LOCAL_SIZE;
			u_vec[index] = tmp;

			if(index >= col_limited + 1)
			{
				tmp_sum_value += tmp * tmp;
			}
		}
		SyncThreads();

		// Find length of vector in A's column with reduction sum function
		pr_data_256[id] = tmp_sum_value;
		SyncThreads();
		parallel_reduction_sum_256(&vec_length, pr_data_256, col_limited + 1);

		// NOTE: GCN Opencl compiler can do some optimization with this because if
		// initially wanted col_limited is used to select wich work-item runs which branch
		// it is slower. However using col produces the same result.
		float r_value;
		if(id < col)
		{
			// Copy u_vec value
			r_value = u_vec[id];
		}
		else if(id == col)
		{
			u_length_squared = vec_length;
			vec_length = Sqrt(vec_length + u_vec[col_limited] * u_vec[col_limited]);
			u_vec[col_limited] -= vec_length;
			u_length_squared += u_vec[col_limited] * u_vec[col_limited];
			// (u_length_squared is now updated length squared)
			r_value = vec_length;
		}
		else if(id > col) //Could have "&& id <  R_EDGE" but this is little bit faster
		{
			// Last values on every column are zeros
			r_value = 0.0f;
		}

		int id_limited = min(id, buffers - 3);
		if(col < buffers - 3)
			store_r_mat_broadcast(r_mat, col_limited, id_limited, r_value);
		else
			store_r_mat_channel(r_mat, col_limited, id_limited, col - buffers + 3, r_value);
		SyncThreads();

		// Transform further columns of A
		// NOTE: three last columns are three color channels of noisy data. However,
		// they all need to be transfomed as they were column indexed (buffers - 3)
		for(int feature_buffer = col_limited+1; feature_buffer < buffers; feature_buffer++)
		{
			// Starts by computing dot product with reduction sum function
			#if CACHE_TMP_DATA
			// No need to load features_buffer twice because each work-item first copies value for
			// dot product computation and then modifies the same value
			float tmp_data_private_cache[(BLOCK_EDGE_LENGTH * BLOCK_EDGE_LENGTH) / LOCAL_SIZE];
			#endif

			float tmp_sum_value = 0.f;
			for(int sub_vector = 0; sub_vector < BLOCK_PIXELS / LOCAL_SIZE; ++sub_vector)
			{
				const int index = id + sub_vector * LOCAL_SIZE;
				if(index >= col_limited)
				{
					// Load feature
					float tmp = LOAD(features_buffer, IN_ACCESS);

					// [Section 3.4] - Stochastic regularization
					// To handle rank-deficiency in the T matrix, add zero-mean noise to the input buffers
					// (the first time values are loaded), which makes them linearly independent.
					// Note: does not add noise to constant buffer (column 0) and noisy image data (last 3 columns).
					if(col == 0 && feature_buffer < buffers - 3)
					{
						tmp = add_random(tmp, id, sub_vector, feature_buffer, frame_number);
					}

					#if CACHE_TMP_DATA
					tmp_data_private_cache[sub_vector] = tmp;
					#endif
					tmp_sum_value += tmp * u_vec[index];
				}
			}

			pr_data_256[id] = tmp_sum_value;
			SyncThreads();
			parallel_reduction_sum_256(&dotProd, pr_data_256, col_limited);

			// LOCAL_SIZE = size of the shared memory (= 256)
			// BLOCK_PIXELS = number of pixels in a block (32x32 = 1024)

			// #define HORIZONTAL_BLOCKS (WORKSET_WIDTH / BLOCK_EDGE_LENGTH)
			// #define BLOCK_INDEX_X (group_id % (HORIZONTAL_BLOCKS + 1))
			// #define BLOCK_INDEX_Y (group_id / (HORIZONTAL_BLOCKS + 1))
			// #define IN_BLOCK_INDEX (BLOCK_INDEX_Y * (HORIZONTAL_BLOCKS + 1) + BLOCK_INDEX_X)
			// #define FEATURE_START (feature_buffer * BLOCK_PIXELS)
			// #define IN_ACCESS (IN_BLOCK_INDEX * buffers * BLOCK_PIXELS + FEATURE_START + sub_vector * LOCAL_SIZE + id)

			// Manual unrolling as the block contains 1024 (32x32) work items and we operate on 256 elements (group size)
			// -> Compute the sum of N values (N = 1024/256 = 4)
			for(int sub_vector = 0; sub_vector < BLOCK_PIXELS / LOCAL_SIZE; ++sub_vector)
			{
				const int index = id + sub_vector * LOCAL_SIZE;
				if (index >= col_limited)
				{
					#if CACHE_TMP_DATA
					float store_value = tmp_data_private_cache[sub_vector];
					#else
					float store_value = LOAD(features_buffer, IN_ACCESS);
					store_value = add_random(store_value, id, sub_vector, feature_buffer, frame_number);
					#endif
					store_value -= 2 * u_vec[index] * dotProd / u_length_squared;
					STORE(features_buffer, IN_ACCESS, store_value);
				}
			}
			SyncThreads();
		}
	}

	// Back substitution
	__shared__ vec3 divider; // Shared memory variable that holds the divider

	// R_EDGE = buffer_count - 2 (= number of features + 3 (noisy color spp buffer) - 2)
	// R is a (M + 1)x(M + 1) matrix, with M the number of features (here equal to buffer_count - 3)
	// which gives us R_EDGE = M + 1 = buffer_count - 3 + 1 = buffer_count - 2
	for(int i = R_EDGE - 2; i >= 0; i--)
	{
		if(id == 0)
			divider = load_r_mat(r_mat, i, i);
		
		SyncThreads();
		
		#if COMPRESSED_R
		if(id < R_EDGE && id >= i)
		#else
		// First values are always zero if R !COMPRESSED_R and
		// "&& id >= i" makes not compressed code run little bit slower
		if(id < R_EDGE)
		#endif
		{
			vec3 value = load_r_mat(r_mat, id, i);
			store_r_mat(r_mat, id, i, value / divider);
		}

		SyncThreads();

		if(id == 0) //Optimization proposal: parallel reduction
		{
			for(int j = i + 1; j < R_EDGE - 1; j++)
			{
				vec3 value  = load_r_mat(r_mat, R_EDGE - 1, i);
				vec3 value2 = load_r_mat(r_mat, j, i);
				store_r_mat(r_mat, R_EDGE - 1, i, value - value2);
			}
		}

		SyncThreads();

		#if COMPRESSED_R
		if(id < R_EDGE && i >= id)
		#else
		if(id < R_EDGE)
		#endif
		{
			vec3 value  = load_r_mat(r_mat, i, id);
			vec3 value2 = load_r_mat(r_mat, R_EDGE - 1, i);
			store_r_mat(r_mat, i, id, value * value2);
		}
		SyncThreads();
	}

	// The features are stored in the first (buffers-3) values: the last 3 contain the noisy 1spp color channels
	if(id < buffers - 3)
	{
		// Store weights
		const int index = group_id * (buffers - 3) + id;
		const vec3 weight = load_r_mat(r_mat, R_EDGE - 1, id);
		store_float3(weights, index, weight);
	}
}


// Weighted sum kernel /////////////////////////////////////////////////////////
// -> outputs the noise-free 1spp color estimate

__global__ void weighted_sum(
	const float * __restrict__ weights,				// [in]	 Features weights computed by the fitter kernel
	const float * __restrict__ mins_maxs,			// [in]  Min and max of features values per block (world_positions)
		  float * __restrict__ output,				// [out] Noise-free color estimate
	const float * __restrict__ current_normals,		// [in]  Current (world) normals
	const float * __restrict__ current_positions,	// [in]  Current world positions
	const float * __restrict__ current_noisy,		// [in]  Current noisy 1spp color (only used for debugging)
	const int frame_number							// [in]  Current frame number
)
{
	// 2D pixel coordinates in [0, IMAGE_WIDTH-1]x[0, IMAGE_HEIGHT-1]
	const ivec2 pixel = ivec2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	
	if(pixel.x >= IMAGE_WIDTH || pixel.y >= IMAGE_HEIGHT)
		return;

	// Linear pixel index
	const int linear_pixel = pixel.y * IMAGE_WIDTH + pixel.x;

	// Load weights and min_max which this pixel should use.
	const ivec2 offset = BLOCK_OFFSETS[frame_number % BLOCK_OFFSETS_COUNT]; // TODO: input directly 'frame_number%BLOCK_OFFSETS_COUNT'

	// Retrieve linear group index from the offset pixel
	// BLOCK_EDGE_HALF = half block size (32/2 -> 16)
	const ivec2 offset_pixel = pixel + BLOCK_EDGE_HALF - offset;
	const int worksetWithMarginBlockCount = (WORKSET_WITH_MARGINS_WIDTH / BLOCK_EDGE_LENGTH); // = worksetBlockCount+1
	const int group_index = (offset_pixel.x / BLOCK_EDGE_LENGTH) + (offset_pixel.y / BLOCK_EDGE_LENGTH) * worksetWithMarginBlockCount;

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

	// !!!!!
	// Uncomment this for debugging. Removes fitting completely.
	//color = load_float3(current_noisy, linear_pixel);

	// Store resutls
	store_float3(output, linear_pixel, color);
}


// Acumulate filtered data kernel //////////////////////////////////////////////
// -> outputs the noise-free accumulated color estimate + a tonemapped version w/ albedo

// TODO: make 2 versions: one for frame 0 and one for the rest (avoid a branch)

__global__ void accumulate_filtered_data(
	const float * __restrict__ filtered_frame,			// [in]  Noise free color estimate (computed as the weighted sum of the features)
	const vec2 * __restrict__ in_prev_frame_pixel,		// [in]  Previous frame pixel coordinates (after reprojection)
	const unsigned char * __restrict__ accept_bools,	// [in]  Validity mask of bilinear samples in previous frame (after reprojection)
	const float * __restrict__ albedo_buffer,			// [in]  Albedo buffer of the current frame (non-noisy)
		  float * __restrict__ tone_mapped_frame,		// [out] Accumulated and tonemapped noise-free color estimate
	const unsigned char* __restrict__ current_spp,		// [in]	 Current number of samples accumulated (for CMA)
	const float * __restrict__ accumulated_prev_frame,	// [in]  Previous frame noise-free accumulated color estimate 
		  float * __restrict__ accumulated_frame,		// [out] Current frame noise-free accumulated color estimate
	const int frame_number								// [in]  Current frame number
)
{
	// 2D pixel coordinates in [0, IMAGE_WIDTH-1]x[0, IMAGE_HEIGHT-1]
	const ivec2 pixel = ivec2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
   
	if(pixel.x >= IMAGE_WIDTH || pixel.y >= IMAGE_HEIGHT)
		return;

	// Linear pixel index
	const int linear_pixel = pixel.y * IMAGE_WIDTH + pixel.x;

	// Noise-free estimate of the color (computed via a weighted sum of features)
	vec3 filtered_color = load_float3(filtered_frame, linear_pixel);
	vec3 prev_color = vec3(0.f, 0.f, 0.f);
	float blend_alpha = 1.f;

	// Reproject and accumulate previous frame noise-free estimate
	//!!!!!!
	// Add "&& false" to debug other kernels (removes accumulation completely)
	if(frame_number > 0)
	{
		// Bitmask telling which bilinear samples were accepted in the first accumulation kernel
		const unsigned char accept = accept_bools[linear_pixel];

		if(accept > 0) // If any prev frame sample is accepted
		{
			// Pixel coordinates in the previous frame (in [0, IMAGE_WIDTH-1]x[0, IMAGE_HEIGHT-1])
			const vec2 prev_frame_pixel_f = in_prev_frame_pixel[linear_pixel];
			
			// Integer pixel coordinates in the previous frame
			const ivec2 prev_frame_pixel_i = FloatToIntRn(prev_frame_pixel_f);

			// Compute bilinear weights for bilinear sampling
			const vec2 prev_pixel_fract = prev_frame_pixel_f - vec2(prev_frame_pixel_i);
			const vec2 one_minus_prev_pixel_fract = 1.f - prev_pixel_fract;
			
			float total_weight = 0.f;

			// Add valid bilinear samples

			if(accept & 0x01)
			{
				float weight = one_minus_prev_pixel_fract.x * one_minus_prev_pixel_fract.y;
				int linear_sample_location = prev_frame_pixel_i.y * IMAGE_WIDTH + prev_frame_pixel_i.x;
				prev_color += weight * load_float3(accumulated_prev_frame, linear_sample_location);
				total_weight += weight;
			}

			if(accept & 0x02)
			{
				float weight = prev_pixel_fract.x * one_minus_prev_pixel_fract.y;
				int linear_sample_location = prev_frame_pixel_i.y * IMAGE_WIDTH + prev_frame_pixel_i.x + 1;
				prev_color += weight * load_float3(accumulated_prev_frame, linear_sample_location);
				total_weight += weight;
			}

			if(accept & 0x04)
			{
				float weight = one_minus_prev_pixel_fract.x * prev_pixel_fract.y;
				int linear_sample_location = (prev_frame_pixel_i.y + 1) * IMAGE_WIDTH + prev_frame_pixel_i.x;
				prev_color += weight * load_float3(accumulated_prev_frame, linear_sample_location);
				total_weight += weight;
			}

			if(accept & 0x08)
			{
				float weight = prev_pixel_fract.x * prev_pixel_fract.y;
				int linear_sample_location = (prev_frame_pixel_i.y + 1) * IMAGE_WIDTH + prev_frame_pixel_i.x + 1;
				prev_color += weight * load_float3(accumulated_prev_frame, linear_sample_location);
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
				blend_alpha = 1.f / convert_float(current_spp[linear_pixel]);
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
   const vec3 tone_mapped_color = Clamp(Pow(Max(0.f, albedo * accumulated_color), 0.454545f), 0.f, 1.f);
   store_float3(tone_mapped_frame, linear_pixel, tone_mapped_color);
}
