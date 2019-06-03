/*  The MIT License (MIT)
 *  
 *  Copyright (c) 2019 Matias Koskela / Tampere University
 *  Copyright (c) 2018 Kalle Immonen / Tampere University of Technology
 *  
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:
 *  
 *  The above copyright notice and this permission notice shall be included in
 *  all copies or substantial portions of the Software.
 *  
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *  THE SOFTWARE.
 */


inline void SyncThreads()
{
	barrier(CLK_LOCAL_MEM_FENCE); // Equivalent to CUDA __syncthreads() (https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/barrier.html)
}

// Unrolled parallel sum reduction of 256 values
// TODO: unused start_index...
static inline void parallel_reduction_sum_256(__local float * result, __local float * pr_data_256, const int start_index)
{
	const int id = get_local_id(0); // = threadIdx.x

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

// Unrolled parallel min reduction of 256 values
static inline void parallel_reduction_min_256(__local float * result, __local float * pr_data_256)
{
	const int id = get_local_id(0); // = threadIdx.x

	if(id < 64)
		pr_data_256[id] = fmin(fmin(fmin(pr_data_256[id], pr_data_256[id + 64]), pr_data_256[id + 128]), pr_data_256[id + 192]);
	SyncThreads();

	if(id < 8)
		pr_data_256[id] = fmin(fmin(fmin(fmin(fmin(fmin(fmin(pr_data_256[id], pr_data_256[id + 8]),
			pr_data_256[id + 16]), pr_data_256[id + 24]), pr_data_256[id + 32]), pr_data_256[id + 40]),
			pr_data_256[id + 48]), pr_data_256[id + 56]);
	SyncThreads();

	if(id == 0)
	{
		*result = fmin(fmin(fmin(fmin(fmin(fmin(fmin(pr_data_256[0], pr_data_256[1]), pr_data_256[2]),
			pr_data_256[3]), pr_data_256[4]), pr_data_256[5]), pr_data_256[6]), pr_data_256[7]);
	}
	SyncThreads();
}

// Unrolled parallel max reduction of 256 values
static inline void parallel_reduction_max_256(__local float * result, __local float * pr_data_256)
{
   const int id = get_local_id(0); // = threadIdx.x

	if(id < 64)
		pr_data_256[id] = fmax(fmax(fmax(pr_data_256[id], pr_data_256[id + 64]), pr_data_256[id + 128]), pr_data_256[id + 192]);
	SyncThreads();

	if(id < 8)
		pr_data_256[id] = fmax(fmax(fmax(fmax(fmax(fmax(fmax(pr_data_256[id], pr_data_256[id + 8]),
			pr_data_256[id + 16]), pr_data_256[id + 24]), pr_data_256[id + 32]), pr_data_256[id + 40]),
			pr_data_256[id + 48]), pr_data_256[id + 56]);
	SyncThreads();

	if(id == 0)
	{
		*result = fmax(fmax(fmax(fmax(fmax(fmax(fmax(pr_data_256[0], pr_data_256[1]), pr_data_256[2]),
			pr_data_256[3]), pr_data_256[4]), pr_data_256[5]), pr_data_256[6]), pr_data_256[7]);
	}
	SyncThreads();
}

// Helper defines used in IN_ACCESS define
#define BLOCK_EDGE_HALF (BLOCK_EDGE_LENGTH / 2)
#define HORIZONTAL_BLOCKS (WORKSET_WIDTH / BLOCK_EDGE_LENGTH)
#define BLOCK_INDEX_X (group_id % (HORIZONTAL_BLOCKS + 1))
#define BLOCK_INDEX_Y (group_id / (HORIZONTAL_BLOCKS + 1))
#define IN_BLOCK_INDEX (BLOCK_INDEX_Y * (HORIZONTAL_BLOCKS + 1) + BLOCK_INDEX_X)
#define FEATURE_START (feature_buffer * BLOCK_PIXELS)
#define IN_ACCESS (IN_BLOCK_INDEX * buffers * BLOCK_PIXELS + FEATURE_START + sub_vector * 256 + id)


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

static inline float3 load_r_mat(__local const float3* r_mat, const int x, const int y)
{
   return r_mat[R_ACCESS];
}

static inline void store_r_mat(__local float3* r_mat, const int x, const int y, const float3 value)
{
   r_mat[R_ACCESS] = value;
}

static inline void store_r_mat_broadcast(__local float3* r_mat, const int x, const int y, const float value)
{
   r_mat[R_ACCESS] = value;
}

static inline void store_r_mat_channel(__local float3* r_mat, const int x, const int y, const int channel, const float value)
{
   if(channel == 0)
      r_mat[R_ACCESS].x = value;
   else if(channel == 1)
      r_mat[R_ACCESS].y = value;
   else // channel == 2
      r_mat[R_ACCESS].z = value;
}

// Random generator from here http://asgerhoedt.dk/?p=323
inline unsigned int thrust_hash(unsigned int seed)
{
	seed = (seed+0x7ed55d16) + (seed<<12);
	seed = (seed^0xc761c23c) ^ (seed>>19);
	seed = (seed+0x165667b1) + (seed<<5);
	seed = (seed+0xd3a2646c) ^ (seed<<9);
	seed = (seed+0xfd7046c5) + (seed<<3);
	seed = (seed^0xb55a4f09) ^ (seed>>16);
	return seed;
}

inline float thrust_rand01(unsigned int seed)
{
	return convert_float(thrust_hash(seed)) / convert_float(UINT_MAX);
}

inline unsigned int wang_hash(unsigned int seed)
{
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}

inline float wang_rand01(unsigned int seed)
{
	return convert_float(wang_hash(seed)) / convert_float(UINT_MAX);
}

static inline float add_random(
	const float value,
	const int id,
	const int sub_vector,
	const int feature_buffer,
	const int frame_number
)
{
	const int seed = id + sub_vector * LOCAL_SIZE + feature_buffer * BLOCK_EDGE_LENGTH * BLOCK_EDGE_LENGTH +
				 frame_number * BUFFER_COUNT * BLOCK_EDGE_LENGTH * BLOCK_EDGE_LENGTH;
	
	float noise01 = thrust_rand01(seed);
	//float noise01 = wang_rand01(seed);
	float signedZeroMeanNoise = 2.f * noise01 - 1.f;
	return value + NOISE_AMOUNT * signedZeroMeanNoise;
}

float3 RGB_to_YCoCg(float3 rgb)
{
	return (float3){
		dot(rgb, (float3){ 1.f, 2.f,  1.f}),
		dot(rgb, (float3){ 2.f, 0.f, -2.f}),
		dot(rgb, (float3){-1.f, 2.f, -1.f})
	};
}

float3 YCoCg_to_RGB(float3 YCoCg)
{
	return (float3){
		dot(YCoCg, (float3){0.25f,  0.25f, -0.25f}),
		dot(YCoCg, (float3){0.25f,    0.f,  0.25f}),
		dot(YCoCg, (float3){0.25f, -0.25f, -0.25f})
	};
}

// TODO: try to scale in [-1, +1] to have the same interval for every feature
static inline float scale(float value, float min, float max)
{
	if(fabs(max - min) > 1.0f)
	{
		return (value - min) / (max - min);
	}
	return value - min;
}

// Alternative scale method
static inline float Scale01(float value, float min, float max)
{
	const float span = max - min;
	return (span > 0.0f) ? (value - min) / span : 0.0f;
}

// Simple mirroring of image index if it is out of bounds.
// NOTE: Works only if index is less than one size out of bounds.
// NOTE: The mirroring duplicate borders: 3 2 1 0 | 0 1 2 3 | 3 2 1 0
static inline int mirror(int index, int size)
{
	if(index < 0)
		index = abs(index) - 1;
	else if(index >= size)
		index = 2 * size - index - 1;

	return index;
}

static inline int2 mirror2(int2 index, int2 size)
{
	index.x = mirror(index.x, size.x);
	index.y = mirror(index.y, size.y);
	return index;
}

static inline void store_float3(__global float* restrict buffer, const int index, const float3 value)
{
	buffer[index * 3 + 0] = value.x;
	buffer[index * 3 + 1] = value.y;
	buffer[index * 3 + 2] = value.z;
}

// This is significantly slower the the inline function on Vega FE
//#define store_float3(buffer, index, value) \
//   buffer[(index) * 3 + 0] = value.x; \
//   buffer[(index) * 3 + 1] = value.y; \
//   buffer[(index) * 3 + 2] = value.z;

#define load_float3(buffer, index) ((float3)\
	{buffer[(index) * 3], buffer[(index) * 3 + 1], buffer[(index) * 3 + 2]})

// This gives on Vega FE warning about breaking the restrict keyword of the kernel
//static inline float3 load_float3(
//   __global float* restrict buffer,
//   const int index){
//
//   return (float3){
//      buffer[index * 3 + 0],
//      buffer[index * 3 + 1],
//      buffer[index * 3 + 2]
//   };
//}

#if USE_HALF_PRECISION_IN_FEATURES_DATA

#define LOAD(buffer, index) vload_half(index, buffer)
#define STORE(buffer, index, value) vstore_half(value, index, buffer)

#else

#define LOAD(buffer, index) buffer[index]
#define STORE(buffer, index, value) buffer[index] = value

#endif

// TODO: try to cycle through all offsets using Bayer matrix
#define BLOCK_OFFSETS_COUNT 16
__constant int2 BLOCK_OFFSETS[BLOCK_OFFSETS_COUNT] = {
	(int2){ -14, -14 },
	(int2){   4,  -6 },
	(int2){  -8,  14 },
	(int2){   8,   0 },
	(int2){ -10,  -8 },
	(int2){   2,  12 },
	(int2){  12, -12 },
	(int2){ -10,   0 },
	(int2){  12,  14 },
	(int2){  -8, -16 },
	(int2){   6,   6 },
	(int2){  -2,  -2 },
	(int2){   6, -14 },
	(int2){ -16,  12 },
	(int2){  14,  -4 },
	(int2){  -6,   4 }
};

// TODO: make two versions of accumulate_noisy_data kernel: one for frame 0 and another for the rest (avoid branch)

// https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/functionQualifiers.html
// The optional __attribute__((reqd_work_group_size(X, Y, Z))) is the work-group size that MUST be used as
// the local_work_size argument to clEnqueueNDRangeKernel.
// This allows the compiler to optimize the generated code appropriately for this kernel.
#if ADD_REQD_WG_SIZE
__attribute__((reqd_work_group_size(LOCAL_WIDTH, LOCAL_HEIGHT, 1)))
#endif
__kernel void accumulate_noisy_data(
	__global float2* restrict out_prev_frame_pixel,			// [out] Previous frame pixel coordinates (after reprojection)
	__global unsigned char* restrict accept_bools,			// [out] Validity mask of bilinear samples in previous frame (after reprojection)
	const __global float* restrict current_normals,			// [in]  Current  (world) normals
	const __global float* restrict previous_normals,		// [in]  Previous (world) normals
	const __global float* restrict current_positions,		// [in]  Current  world positions
	const __global float* restrict previous_positions,		// [in]  Previous world positions
		  __global float* restrict current_noisy,			// [out] Current  noisy 1spp color
	const __global float* restrict previous_noisy,			// [in]  Previous noisy 1spp color
	const __global unsigned char* restrict previous_spp,	// [in]  Previous number of samples accumulated (for CMA)
		  __global unsigned char* restrict current_spp,		// [out] Current  number of samples accumulated (for CMA)
	#if USE_HALF_PRECISION_IN_FEATURES_DATA
	__global half* restrict features_data,					// [out] Features buffer (half-precision)
	#else
	__global float* restrict features_data,					// [out] Features buffer (single-precision)
	#endif
      const float16 prev_frame_camera_matrix,				// [in]  ViewProj matrix of previous frame
      const float2 pixel_offset,
      const int frame_number								// [in]  Current frame number
)
{
	// CUDA equivalent:
	// get_global_id(0) <=> blockIdx.x * blockDim.x + threadIdx.x
	// get_global_id(1) <=> blockIdx.y * blockDim.y + threadIdx.y
	const int2 gid = {get_global_id(0), get_global_id(1)};

	// Mirror indexed of the input. x and y are always less than one size out of
	// bounds if image dimensions are bigger than BLOCK_EDGE_LENGTH
	// BLOCK_EDGE_HALF = half block size (32/2 -> 16)
	const int2 pixel_without_mirror = gid - BLOCK_EDGE_HALF + BLOCK_OFFSETS[frame_number % BLOCK_OFFSETS_COUNT]; // TODO: input directly frame_number % BLOCK_OFFSETS_COUNT

	// Pixel coordinates in [0, IMAGE_WIDTH-1]x[0, IMAGE_HEIGHT-1]
	const int2 pixel = mirror2(pixel_without_mirror, (int2){IMAGE_WIDTH, IMAGE_HEIGHT});

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
	float3 current_color = load_float3(current_noisy, linear_pixel);

	// Current frame world position
	float4 world_position = (float4){0.f, 0.f, 0.f, 1.f};
	world_position.xyz = load_float3(current_positions, linear_pixel);

	// Current frame (world) normal
	float3 normal = load_float3(current_normals, linear_pixel);


	// Previous frame pixel coordinates in [0, IMAGE_WIDTH-1]x[0, IMAGE_HEIGHT-1]
	// Default is the same pixel coordinates (no movement)
	float2 prev_frame_pixel_f = convert_float2(pixel);

	// Bit mask telling which previous frame (bilinear) samples are valid under reprojection into current frame
	unsigned char store_accept = 0x00;

	// Blending factor with history buffer
	// Blend_alpha 1.f means that only current frame color is used. The value is changed if sample from previous frame can be used
	float blend_alpha = 1.f;
	float3 previous_color = (float3){0.f, 0.f, 0.f};

	float sample_spp = 0.f;
	if(frame_number > 0)
	{
		// Project current world position into previous frame with the previous ViewProj matrix

		// Matrix multiplication and normalization to 0..1
		// TODO: transpose camera matrix somewhere else if it hits the performance
		// NOTE: not enough to test performance by changing s048c to s0123 here because it
		// produces prev_frame_pixels outside screen and removes many memory accesses
		float2 prev_frame_uv;
		prev_frame_uv.x = dot(prev_frame_camera_matrix.s048c, world_position); // Transform x
		prev_frame_uv.y = dot(prev_frame_camera_matrix.s159d, world_position); // Transform y
		// No need for z-buffer in accumulation of the noisy data
		// -> might be useful if we use it to detect disocclusion
		// --> compare previous z (store previous frame Z-buffer) with prev_frame_pixel_uv.z
		//prev_frame_uv.z = dot(prev_frame_camera_matrix.s26ae, world_position); // Transform z
		prev_frame_uv /= dot(prev_frame_camera_matrix.s37bf, world_position);
		prev_frame_uv += 1.f; prev_frame_uv /= 2.f; // prev_frame_uv = prev_frame_uv * 0.5f + 0.5f;

		// Compute the pixel coordinates in the previous frame (in [0, IMAGE_WIDTH-1]x[0, IMAGE_HEIGHT-1])
		prev_frame_pixel_f = prev_frame_uv * (float2){IMAGE_WIDTH, IMAGE_HEIGHT};

		// Apply offset (TODO: what offset??? seems to always be 0.5... Maybe TAA/ray subpixel offsets -> send 1-pixel_offset.y)
		prev_frame_pixel_f -= (float2){pixel_offset.x, 1 - pixel_offset.y};

		// Convert in to integer pixel coordinates (round to negative infinity)
		int2 prev_frame_pixel_i = convert_int2_rtn(prev_frame_pixel_f);

		// Compute bilinear weights (for bilinear sampling)
		// TODO: implement bicubic Catmull-Rom (for sharpness)? => would need to perform more fetches and store more "validity bits" in mask
		int2 offsets[4];
		offsets[0] = (int2){0, 0};
		offsets[1] = (int2){1, 0};
		offsets[2] = (int2){0, 1};
		offsets[3] = (int2){1, 1};
		float2 prev_pixel_fract = prev_frame_pixel_f - convert_float2(prev_frame_pixel_i);
		float2 one_minus_prev_pixel_fract = 1.f - prev_pixel_fract;
		float weights[4];
		weights[0] = one_minus_prev_pixel_fract.x * one_minus_prev_pixel_fract.y;
		weights[1] = prev_pixel_fract.x           * one_minus_prev_pixel_fract.y;
		weights[2] = one_minus_prev_pixel_fract.x * prev_pixel_fract.y;
		weights[3] = prev_pixel_fract.x           * prev_pixel_fract.y;
		float total_weight = 0.f;

		// Bilinear sampling
		for(int i = 0; i < 4; ++i)
		{
			int2 sample_location = prev_frame_pixel_i + offsets[i];
			int linear_sample_location = sample_location.y * IMAGE_WIDTH + sample_location.x;

			// Check if previous frame color can be used based on its screen location
			if(sample_location.x >= 0 && sample_location.y >= 0 &&
			   sample_location.x < IMAGE_WIDTH && sample_location.y < IMAGE_HEIGHT
			)
			{
				// Fetch previous frame world position
				float3 prev_world_position = load_float3(previous_positions, linear_sample_location);

				// Compute world distance squared
				float3 position_difference = prev_world_position - world_position.xyz;
				float position_distance_squared = dot(position_difference, position_difference);

				// World position distance discard
				if(position_distance_squared < convert_float(POSITION_LIMIT_SQUARED)){

					// Fetch previous frame normal
					float3 prev_normal = load_float3(previous_normals, linear_sample_location);

					// Distance of the normals
					// NOTE: could use some other distance metric (e.g. angle), but we use hard
					// experimentally found threshold -> means that the metric doesn't matter.
					float3 normal_difference = prev_normal - normal;
					float normal_distance_squared = dot(normal_difference, normal_difference);

					if(normal_distance_squared < convert_float(NORMAL_LIMIT_SQUARED)){

						// Pixel passes all tests so store it to "validity bitmask"
						store_accept |= 1 << i;

						// Accumulate number of samples
						sample_spp += weights[i] * convert_float(previous_spp[linear_sample_location]);

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
			blend_alpha = fmax(blend_alpha, BLEND_ALPHA);
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
		// _sat is just extra causion because sample_spp should be less equal than 254
		new_spp = (sample_spp > 254.f) ? 255 : convert_uchar_sat_rte(sample_spp) + 1;
	}
	current_spp[linear_pixel] = new_spp; // Store current number of samples accumulated (for CMA)

	float3 new_color = blend_alpha * current_color + (1.f - blend_alpha) * previous_color; // Lerp(previous_color, current_color, blend_alpha);


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
		feature = max(min(feature, 65504.f), -65504.f); // Clamp(feature, -65504.f, 65504.f);
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
		//current_spp[linear_pixel] = new_spp; // Store current number of samples accumulated (for CMA)
	}
}

// TODO: add an extra pass to normalize features and copy them to a single buffer
// -> normalize world_position only once and then compute world_position*world_position
// --> avoid parallel reduction for higher order features
// --> avoid output min/max that is used to scale the scaled features in the kernel "weighted_sum"
//	   (there seems to be double scaling: once in fitter and once weighted_sum...)
//
// OR directly generate a normalized world_position by dividing by the bbox of the scene or some value + saturate
// -> allow to skip parallel min/max reductions altogether

// Note: The __local or local address space name is used to describe variables that need to be allocated in local memory
// and are shared by all work-items of a work-group. (https://www.khronos.org/registry/OpenCL/sdk/1.1/docs/man/xhtml/local.html)

#if ADD_REQD_WG_SIZE
__attribute__((reqd_work_group_size(256, 1, 1)))
#endif
__kernel void fitter(
	__local  float*  pr_data_256,					// [local] Shared memory used to perform parrallel reduction (max, min, sum)
	__local  float*  u_vec,							// [local] Shared memory used to store the 'u' vectors
	__local  float3* r_mat,							// [local] Shared memory used to store the R matrix of the QR factorization
	__global float* restrict weights,				// [out]   Features weights
	__global float* restrict mins_maxs,				// [out]   Min and max of features values per block (world_positions)
	#if USE_HALF_PRECISION_IN_FEATURES_DATA
	__global half* restrict tmp_data,				// [out]   Features buffer (half-precision)
	#else
	__global float* restrict tmp_data,				// [out]   Features buffer (single-precision)
	#endif
	const int frame_number							// [in]	   Current frame number
)
{
	__local float u_length_squared;					// Local variable (i.e shared across threads in group) that holds the 'u' vector square length
	__local float dot;								// Local variable (i.e shared across threads in group) that holds the 
	__local float block_min;						// Local variable (i.e shared across threads in group) that holds the result of the parallel min reduction
	__local float block_max;						// Local variable (i.e shared across threads in group) that holds the result of the parallel max reduction
	__local float vec_length;			

	const int group_id = get_group_id(0); // = blockIdx.x
	const int id = get_local_id(0); // = threadIdx.x in [0, 255]
	const int buffers = BUFFER_COUNT;

	// Scales world positions to 0..1 in a block
	const int feature_to_scale_beg_idx = FEATURES_NOT_SCALED;
	const int feature_to_scale_end_idx = buffers - 3;
	for(int feature_buffer = feature_to_scale_beg_idx; feature_buffer < feature_to_scale_end_idx; ++feature_buffer)
	{
		// Find maximum and minimum of the whole block
		float tmp_max = -INFINITY;
		float tmp_min = +INFINITY;
	  
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
			float value = LOAD(tmp_data, IN_ACCESS);
			tmp_max = fmax(value, tmp_max);
			tmp_min = fmin(value, tmp_min);
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
		SyncThreads();

		// Scale feature and replace value in features buffer
		for(int sub_vector = 0; sub_vector < BLOCK_PIXELS / LOCAL_SIZE; ++sub_vector)
		{
			float scaled_value = scale(LOAD(tmp_data, IN_ACCESS), block_min, block_max);
			STORE(tmp_data, IN_ACCESS, scaled_value);
		}
	}

	// Non square matrices require processing every column. Otherwise result is
	// OKish, but R is not upper triangular matrix
	int limit = buffers == BLOCK_PIXELS ? buffers - 1 : buffers;

	// Compute R
	for(int col = 0; col < limit; col++)
	{
		// Note: the last 3 features values are the 3 channels of the color (not used for the regression)
		int col_limited = min(col, buffers - 3);

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
			float tmp = LOAD(tmp_data, IN_ACCESS);

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
			vec_length = sqrt(vec_length + u_vec[col_limited] * u_vec[col_limited]);
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
			// No need to load tmp_data twice because each work-item first copies value for
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
					float tmp = LOAD(tmp_data, IN_ACCESS);

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
			parallel_reduction_sum_256(&dot, pr_data_256, col_limited);

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
					float store_value = LOAD(tmp_data, IN_ACCESS);
					store_value = add_random(store_value, id, sub_vector, feature_buffer, frame_number);
					#endif
					store_value -= 2 * u_vec[index] * dot / u_length_squared;
					STORE(tmp_data, IN_ACCESS, store_value);
				}
			}
			SyncThreads();
		}
	}

	// Back substitution
	__local float3 divider; // Local variable (i.e shared across threads in group) that holds the divider

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
			float3 value = load_r_mat(r_mat, id, i);
			store_r_mat(r_mat, id, i, value / divider);
		}

		SyncThreads();

		if(id == 0) //Optimization proposal: parallel reduction
		{
			for(int j = i + 1; j < R_EDGE - 1; j++)
			{
				float3 value  = load_r_mat(r_mat, R_EDGE - 1, i);
				float3 value2 = load_r_mat(r_mat, j, i);
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
			float3 value  = load_r_mat(r_mat, i, id);
			float3 value2 = load_r_mat(r_mat, R_EDGE - 1, i);
			store_r_mat(r_mat, i, id, value * value2);
		}
		SyncThreads();
	}

	// The features are stored in the first (buffers-3) values: the last 3 contain the noisy 1spp color channels
	if(id < buffers - 3)
	{
		// Store weights
		const int index = group_id * (buffers - 3) + id;
		const float3 weight = load_r_mat(r_mat, R_EDGE - 1, id);
		store_float3(weights, index, weight);
	}
}


__kernel void weighted_sum(
	const __global float* restrict weights,				// [in]	 Features weights computed by the fitter kernel
	const __global float* restrict mins_maxs,			// [in]  Min and max of features values per block (world_positions)
		  __global float* restrict output,				// [out] Noise-free color estimate
	const __global float* restrict current_normals,		// [in]  Current (world) normals
	const __global float* restrict current_positions,	// [in]  Current world positions
	const __global float* restrict current_noisy,		// [in]  Current noisy 1spp color (only used for debugging)
	const int frame_number								// [in]  Current frame number
)
{
	// CUDA equivalent:
	// get_global_id(0) <=> blockIdx.x * blockDim.x + threadIdx.x
	// get_global_id(1) <=> blockIdx.y * blockDim.y + threadIdx.y
	// 2D pixel coordinates in [0, IMAGE_WIDTH-1]x[0, IMAGE_HEIGHT-1]
	const int2 pixel = (int2){get_global_id(0), get_global_id(1)};
	
	if(pixel.x >= IMAGE_WIDTH || pixel.y >= IMAGE_HEIGHT)
		return;

	// Linear pixel index
	const int linear_pixel = pixel.y * IMAGE_WIDTH + pixel.x;

	// Load weights and min_max which this pixel should use.
	const int2 offset = BLOCK_OFFSETS[frame_number % BLOCK_OFFSETS_COUNT]; // TODO: input directly 'frame_number%BLOCK_OFFSETS_COUNT'

	// Retrieve linear group index from the offset pixel
	// BLOCK_EDGE_HALF = half block size (32/2 -> 16)
	const int2 offset_pixel = pixel + BLOCK_EDGE_HALF - offset;
	const int group_index = (offset_pixel.x / BLOCK_EDGE_LENGTH) + (offset_pixel.y / BLOCK_EDGE_LENGTH) * (WORKSET_WITH_MARGINS_WIDTH / BLOCK_EDGE_LENGTH);

	// Reload features from buffer here to have values without stochastic regularization noise
	// TODO: bind the normalized world_position buffer to avoid renormalizing again (no need for mins_maxs buffer)
	float3 world_position = load_float3(current_positions, linear_pixel); 
	float3 normal = load_float3(current_normals, linear_pixel);
	float features[BUFFER_COUNT - 3] =
	{
		// TODO: replace with function that fill the array?
		FEATURE_BUFFERS // expands to 1.f, normal.x, ..., world_position.x, ..., world_position.x * world_position.x, ...
	};

	// Weighted sum of the feature buffers
	float3 color = (float3){0.f, 0.f, 0.f};
	for(int feature_buffer = 0; feature_buffer < BUFFER_COUNT - 3; feature_buffer++)
	{
		float feature = features[feature_buffer];

		// Scale world position buffers
		if (feature_buffer >= FEATURES_NOT_SCALED)
		{
			const int min_max_index = (group_index * FEATURES_SCALED + feature_buffer - FEATURES_NOT_SCALED) * 2;
			feature = scale(feature, mins_maxs[min_max_index + 0], mins_maxs[min_max_index + 1]);
		}

		// Load weight and sum
		float3 weight = load_float3(weights, group_index * (BUFFER_COUNT - 3) + feature_buffer);
		color += weight * feature;
	}

   // Remove negative values from every component of the fitting results
   color = color < 0.f ? 0.f : color;

   // !!!!!
   // Uncomment this for debugging. Removes fitting completely.
   //color = load_float3(current_noisy, linear_pixel);

   // Store resutls
   store_float3(output, linear_pixel, color);
}


// TODO: make 2 versions: one for frame 0 and one for the rest (avoid a branch)

__kernel void accumulate_filtered_data(
	const __global float* restrict filtered_frame,			// [in]  Noise free color estimate (computed as the weighted sum of the features)
	const __global float2* restrict in_prev_frame_pixel,	// [in]  Previous frame pixel coordinates (after reprojection)
	const __global unsigned char* restrict accept_bools,	// [in]  Validity mask of bilinear samples in previous frame (after reprojection)
	const __global float* restrict albedo,					// [in]  Albedo buffer of the current frame (non-noisy)
		  __global float* restrict tone_mapped_frame,		// [out] Accumulated and tonemapped noise-free color estimate
	const __global unsigned char* restrict current_spp,		// [in]	 Current number of samples accumulated (for CMA)
	const __global float* restrict accumulated_prev_frame,	// [in]  Previous frame noise-free accumulated color estimate 
		  __global float* restrict accumulated_frame,		// [out] Current frame noise-free accumulated color estimate
	const int frame_number									// [in]  Current frame number
)
{
	// CUDA equivalent:
	// get_global_id(0) <=> blockIdx.x * blockDim.x + threadIdx.x
	// get_global_id(1) <=> blockIdx.y * blockDim.y + threadIdx.y
	// 2D pixel coordinates in [0, IMAGE_WIDTH-1]x[0, IMAGE_HEIGHT-1]
	const int2 pixel = (int2){get_global_id(0), get_global_id(1)};
   
	if(pixel.x >= IMAGE_WIDTH || pixel.y >= IMAGE_HEIGHT)
		return;

	// Linear pixel index
	const int linear_pixel = pixel.y * IMAGE_WIDTH + pixel.x;

	// Noise-free estimate of the color (computed via a weighted sum of features)
	float3 filtered_color = load_float3(filtered_frame, linear_pixel);
	float3 prev_color = (float3){0.f, 0.f, 0.f};
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
			const float2 prev_frame_pixel_f = in_prev_frame_pixel[linear_pixel];
			
			// Integer pixel coordinates in the previous frame
			const int2 prev_frame_pixel_i = convert_int2_rtn(prev_frame_pixel_f);

			// Compute bilinear weights for bilinear sampling
			const float2 prev_pixel_fract = prev_frame_pixel_f - convert_float2(prev_frame_pixel_i);
			const float2 one_minus_prev_pixel_fract = 1.f - prev_pixel_fract;
			
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
				blend_alpha = 1.f / convert_float(current_spp[linear_pixel]);
				blend_alpha = fmax(blend_alpha, SECOND_BLEND_ALPHA);
				prev_color /= total_weight;

				// Note: we accumulate at most 255 samples for the cumulative moving average (which is more than enough because of
				// the threshold SECOND_BLEND_ALPHA that switch to exponential moving average).
				// E.g: SECOND_BLEND_ALPHA = 0.1 = 1 / (n + 1) <=> n = (1 - 0.1) / 0.1 = 9 => above 9 samples for a pixel,
				// we switch to exponential moving average with alpha = 10%
			}
		}
	}

   // Mix with colors and store results
   float3 accumulated_color = blend_alpha * filtered_color + (1.f - blend_alpha) * prev_color; // Lerp(prev_color, filtered_color, blend_alpha);
   store_float3(accumulated_frame, linear_pixel, accumulated_color);

   // Remodulate albedo and tone map
   float3 my_albedo = load_float3(albedo, linear_pixel);
   const float3 tone_mapped_color = clamp(powr(max(0.f, my_albedo * accumulated_color), 0.454545f), 0.f, 1.f);
   store_float3(tone_mapped_frame, linear_pixel, tone_mapped_color);
}


// TODO:
// - make two versions of the kernel: one for the frame 0 and one for the rest
// - optimize with local/shared memory
__kernel void taa(
	const __global float2* restrict in_prev_frame_pixel,	// [in]  Previous frame pixel coordinates (after reprojection)
	const __global float* restrict new_frame,				// [in]	 Current frame color buffer
		  __global float* restrict result_frame,			// [out] Antialiased frame color buffer
	const __global float* restrict prev_frame,				// [in]  Previous frame color buffer
	const int frame_number									// [in]  Current frame number
)
{
   	// CUDA equivalent:
	// get_global_id(0) <=> blockIdx.x * blockDim.x + threadIdx.x
	// get_global_id(1) <=> blockIdx.y * blockDim.y + threadIdx.y
	// 2D pixel coordinates in [0, IMAGE_WIDTH-1]x[0, IMAGE_HEIGHT-1]
	const int2 pixel = (int2){get_global_id(0), get_global_id(1)};
   
	if(pixel.x >= IMAGE_WIDTH || pixel.y >= IMAGE_HEIGHT)
		return;

	// Linear pixel index
	const int linear_pixel = pixel.y * IMAGE_WIDTH + pixel.x;

	// Current frame color
	float3 my_new_color = load_float3(new_frame, linear_pixel);

	// Previous frame pixel coordinates
	const float2 prev_frame_pixel_f = in_prev_frame_pixel[linear_pixel];
	int2 prev_frame_pixel_i = convert_int2_rtn(prev_frame_pixel_f);

	//!!!!!!
	// Add "|| true" to debug other kernels (removes taa)
	// Return if all sampled pixels are going to be out of image area
	if(frame_number == 0 ||
		prev_frame_pixel_i.x < -1 || prev_frame_pixel_i.y < -1 ||
		prev_frame_pixel_i.x >= IMAGE_WIDTH || prev_frame_pixel_i.y >= IMAGE_HEIGHT
	)
	{
		store_float3(result_frame, linear_pixel, my_new_color);
		return;
	}

	// Compute the color AABB in the 3x3 neighbourhood and the min/max in a cross pattern around the current pixel
	float3 minimum_box = INFINITY;
	float3 minimum_cross = INFINITY;
	float3 maximum_box = -INFINITY;
	float3 maximum_cross = -INFINITY;
	for(int y = -1; y <= 1; ++y)
	{
		for(int x = -1; x <= 1; ++x)
		{
			int2 sample_location = pixel + (int2){x, y};
			if(sample_location.x >= 0 && sample_location.y >= 0 &&
			   sample_location.x < IMAGE_WIDTH && sample_location.y < IMAGE_HEIGHT
			)
			{
				float3 sample_color;
				if(x == 0 && y == 0)
					sample_color = my_new_color;
				else
					sample_color = load_float3(new_frame, sample_location.x + sample_location.y * IMAGE_WIDTH);

				sample_color = RGB_to_YCoCg(sample_color);

				if(x == 0 || y == 0)
				{
					minimum_cross = fmin(minimum_cross, sample_color);
					maximum_cross = fmax(maximum_cross, sample_color);
				}

				minimum_box = fmin(minimum_box, sample_color);
				maximum_box = fmax(maximum_box, sample_color);
			}
		}
	}

	// Bilinear sampling of previous frame.
	// Note: work-item has already returned if the sampling location is completly out of image
	float3 prev_color = (float3){0.f, 0.f, 0.f};
	float total_weight = 0;
	float2 pixel_fract = prev_frame_pixel_f - convert_float2(prev_frame_pixel_i);
	float2 one_minus_pixel_fract = 1.f - pixel_fract;

	if(prev_frame_pixel_i.y >= 0)
	{
		if(prev_frame_pixel_i.x >= 0)
		{
			float weight = one_minus_pixel_fract.x * one_minus_pixel_fract.y;
			prev_color += weight * load_float3(prev_frame, prev_frame_pixel_i.y * IMAGE_WIDTH + prev_frame_pixel_i.x);
			total_weight += weight;
		}

		if(prev_frame_pixel_i.x < IMAGE_WIDTH - 1)
		{
			float weight = pixel_fract.x * one_minus_pixel_fract.y;
			prev_color += weight * load_float3(prev_frame, prev_frame_pixel_i.y * IMAGE_WIDTH + prev_frame_pixel_i.x + 1);
			total_weight += weight;
		}
	}

	if(prev_frame_pixel_i.y < IMAGE_HEIGHT - 1)
	{
		if(prev_frame_pixel_i.x >= 0)
		{
			float weight = one_minus_pixel_fract.x * pixel_fract.y;
			prev_color += weight * load_float3(prev_frame, (prev_frame_pixel_i.y + 1) * IMAGE_WIDTH + prev_frame_pixel_i.x);
			total_weight += weight;
		}

		if(prev_frame_pixel_i.x < IMAGE_WIDTH - 1)
		{
			float weight = pixel_fract.x * pixel_fract.y;
			prev_color += weight * load_float3(prev_frame, (prev_frame_pixel_i.y + 1) * IMAGE_WIDTH + prev_frame_pixel_i.x + 1);
			total_weight += weight;
		}
	}

	prev_color /= total_weight; // Total weight can be less than one on the edges
	float3 prev_color_ycocg = RGB_to_YCoCg(prev_color);

	// Note: Some references use more complicated methods to move the previous frame color to the YCoCg space AABB
	float3 minimum = (minimum_box + minimum_cross) / 2.f;
	float3 maximum = (maximum_box + maximum_cross) / 2.f;
	float3 prev_color_rgb = YCoCg_to_RGB(clamp(prev_color_ycocg, minimum, maximum));

	float3 result_color = TAA_BLEND_ALPHA * my_new_color + (1.f - TAA_BLEND_ALPHA) * prev_color_rgb; // Lerp(prev_color_rgb, my_new_color, TAA_BLEND_ALPHA);
	store_float3(result_frame, linear_pixel, result_color);
}
