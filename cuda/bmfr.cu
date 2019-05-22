

inline void SyncThreads()
{
	__syncthreads();
}

// Unrolled parallel sum reduction of 256 values
// TODO: unused start_index...
inline void parallel_reduction_sum_256(float * result, volatile float * pr_data_256, const int start_index)
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
inline void parallel_reduction_min_256(float * result, volatile float * pr_data_256)
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
inline void parallel_reduction_max_256(float * result, volatile float * pr_data_256)
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

// TODO: change these either by macro that take parameters or inline functions
// Helper defines used in IN_ACCESS define
#define BLOCK_EDGE_HALF (BLOCK_EDGE_LENGTH / 2)
#define HORIZONTAL_BLOCKS (WORKSET_WIDTH / BLOCK_EDGE_LENGTH)
#define BLOCK_INDEX_X (group_id % (HORIZONTAL_BLOCKS + 1))
#define BLOCK_INDEX_Y (group_id / (HORIZONTAL_BLOCKS + 1))
#define IN_BLOCK_INDEX (BLOCK_INDEX_Y * (HORIZONTAL_BLOCKS + 1) + BLOCK_INDEX_X)
#define FEATURE_START (feature_buffer * BLOCK_PIXELS)
#define IN_ACCESS (IN_BLOCK_INDEX * buffers * BLOCK_PIXELS + FEATURE_START + sub_vector * 256 + id)
