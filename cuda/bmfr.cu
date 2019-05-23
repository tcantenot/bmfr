

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


// Load/store float3 functions /////////////////////////////////////////////////

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
#define STORE(buffer, index, value) buffer[index] = __float2half(value)
#else
#define LOAD(buffer, index) buffer[index]
#define STORE(buffer, index, value) buffer[index] = value
#endif
