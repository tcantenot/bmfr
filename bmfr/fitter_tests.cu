#include "fitter_tests.cuh"
#include "new_bmfr.cuh"

using uint = unsigned int;

// https://www.shadertoy.com/view/4tXyWN
inline __device__ float Hash21(uint x, uint y)
{
    uint qx = 1103515245U * ((x >> 1U) ^ y);
    uint qy = 1103515245U * ((y >> 1U) ^ x);
    uint  n = 1103515245U * ((qx) ^ (qy >> 3U));
    return float(n) * (1.0/float(0xffffffffU));
}

inline __device__ void ComputeTestWeights(ivec2 blockIndex, vec3 * weights, uint N)
{
	for(uint i = 0; i < N; ++i)
	{
		float rx = Hash21(blockIndex.x * N + i, blockIndex.y);
		float ry = Hash21(blockIndex.x, blockIndex.y * N + i);
		float rz = Hash21(blockIndex.x * N + i, blockIndex.y * N + i);
		weights[i] = vec3(rx, ry, rz);
	}
}

inline __device__ vec3 ComputeTargetFromTestFeatures(ivec2 blockIndex, float * features)
{
	const uint N = 10;

	vec3 weights[N];
	ComputeTestWeights(blockIndex, weights, N);

	vec3 target = vec3(0);
	for(uint i = 0; i < N; ++i)
	{
		target += weights[i] * features[i];
	}

	return target;
}

inline __device__ void ComputeTestFeatures(ivec2 blockIndex, ivec2 gtid, float * features)
{
	features[0] = 1.f;
	features[1] = Hash21(gtid.x * 3 + 0, gtid.y) * 2.f - 1.f;
	features[2] = Hash21(gtid.x * 3 + 1, gtid.y) * 2.f - 1.f;
	features[3] = Hash21(gtid.x * 3 + 2, gtid.y) * 2.f - 1.f;
	features[4] = Hash21(gtid.x, gtid.y * 3 + 0);
	features[5] = Hash21(gtid.x, gtid.y * 3 + 1);
	features[6] = Hash21(gtid.x, gtid.y * 3 + 2);
	features[7] = features[4] * features[4];
	features[8] = features[5] * features[5];
	features[9] = features[6] * features[6];

	vec3 target = ComputeTargetFromTestFeatures(blockIndex, features);
	features[10] = target.x;
	features[11] = target.y;
	features[12] = target.z;
}

template <int FitterBlockSize>
__global__ void compute_tests_features(
	ComputeTestsFeaturesKernelParams params,
	float * K_RESTRICT features_data // [out] Features buffer
)
{
	const ivec2 gtid = ivec2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

	const int w = params.sizeX;
	const int h = params.sizeY;

	// Mirror indexed of the input. x and y are always less than one size out of bounds if image dimensions are bigger than block size
	const ivec2 pixel_without_mirror = PixelCoordsToShiftedPixelCoords<FitterBlockSize>(gtid, params.frameNumber);

	// Pixel coordinates in [0, w-1]x[0, h-1]
	const ivec2 pixel = mirror2(pixel_without_mirror, ivec2(w, h));

	const uint fitterBlockX = gtid.x / FitterBlockSize;
	const uint fitterBlockY = gtid.y / FitterBlockSize;
	const uint xInFitterBlock = gtid.x % FitterBlockSize;
	const uint yInFitterBlock = gtid.y % FitterBlockSize;

	float features[BUFFER_COUNT];
	ComputeTestFeatures(ivec2(fitterBlockX, fitterBlockY), pixel, features);

	const uint numPixelsInBlock = FitterBlockSize * FitterBlockSize;
	const uint numFeaturesInBlock = numPixelsInBlock * BUFFER_COUNT;
	const uint featuresBaseOffset = fitterBlockY * params.worksetWithMarginBlockCountX * numFeaturesInBlock +
									fitterBlockX * numFeaturesInBlock +
									yInFitterBlock * FitterBlockSize + xInFitterBlock;
	
	for(uint featureIndex = 0; featureIndex < BUFFER_COUNT; ++featureIndex)
	{
		const uint featureOffset = featuresBaseOffset + featureIndex * numPixelsInBlock;
		features_data[featureOffset] = features[featureIndex];
	}
}



extern "C" void run_compute_tests_features(
	dim3 const & grid_size,
	dim3 const & block_size,
	ComputeTestsFeaturesKernelParams const & params,
	float * K_RESTRICT features_data // [out] Features buffer
)
{
	switch(params.fitterBlockSize)
	{
		case 16: compute_tests_features<16><<<grid_size, block_size>>>(params, features_data); break;
		case 32: compute_tests_features<32><<<grid_size, block_size>>>(params, features_data); break;
		case 64: compute_tests_features<64><<<grid_size, block_size>>>(params, features_data); break;
		default: break;
	}
}


template <int FitterBlockSize>
__global__ void test_weighted_sum(
	WeightedSumTestKernelParams params,
	const float * K_RESTRICT weights,	// [in] Features weights computed by the fitter kernel
	float * K_RESTRICT outputs,
	float * K_RESTRICT refs
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
	const ivec2 offset_pixel = ShiftedPixelCoordsToPixelCoords<FitterBlockSize>(pixel, params.frameNumber);
	const uint fitterBlockX = (offset_pixel.x / FitterBlockSize);
	const uint fitterBlockY = (offset_pixel.y / FitterBlockSize);
	const uint fitterBlockIdx = fitterBlockY * params.worksetWithMarginBlockCountX + fitterBlockX;

	float features[BUFFER_COUNT];
	ComputeTestFeatures(ivec2(fitterBlockX, fitterBlockY), pixel, features);

	const unsigned int baseWeightOffset = fitterBlockIdx * (BUFFER_COUNT - 3);

	// Weighted sum of the feature buffers
	vec3 output = vec3(0.f, 0.f, 0.f);
	for(int feature_buffer = 0; feature_buffer < BUFFER_COUNT - 3; feature_buffer++)
	{
		float feature = features[feature_buffer];
		vec3 weight = vec3(
						weights[(baseWeightOffset + feature_buffer) * 3 + 0],
						weights[(baseWeightOffset + feature_buffer) * 3 + 1],
						weights[(baseWeightOffset + feature_buffer) * 3 + 2]);


		output += weight * feature;
	}

	outputs[linear_pixel * 3 + 0] = output.x;
	outputs[linear_pixel * 3 + 1] = output.y;
	outputs[linear_pixel * 3 + 2] = output.z;
	refs[linear_pixel * 3 + 0] = features[10];
	refs[linear_pixel * 3 + 1] = features[11];
	refs[linear_pixel * 3 + 2] = features[12];
}


extern "C" void run_test_weighted_sum(
	dim3 const & grid_size,
	dim3 const & block_size,
	WeightedSumTestKernelParams const & params,
	const float * K_RESTRICT weights,			// [in]	 Features weights computed by the fitter kernel
	float * K_RESTRICT outputs,
	float * K_RESTRICT refs
)
{
	switch(params.fitterBlockSize)
	{
		case 16: test_weighted_sum<16><<<grid_size, block_size>>>(params, weights, outputs, refs); break;
		case 32: test_weighted_sum<32><<<grid_size, block_size>>>(params, weights, outputs, refs); break;
		case 64: test_weighted_sum<64><<<grid_size, block_size>>>(params, weights, outputs, refs); break;
		default: break;
	}
}
