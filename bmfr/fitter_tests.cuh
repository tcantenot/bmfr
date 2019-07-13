#pragma once

#include "config.cuh"
#include "math.cuh"

struct ComputeTestsFeaturesKernelParams
{
	unsigned int sizeX;
	unsigned int sizeY;
	unsigned int fitterBlockSize;
	unsigned int worksetWithMarginBlockCountX;
	unsigned int frameNumber;
};

extern "C" void run_compute_tests_features(
	dim3 const & grid_size,
	dim3 const & block_size,
	ComputeTestsFeaturesKernelParams const & params,
	float * K_RESTRICT features_data // [out] Features buffer
);


struct WeightedSumTestKernelParams
{
	unsigned int sizeX;
	unsigned int sizeY;
	unsigned int fitterBlockSize;
	unsigned int worksetWithMarginBlockCountX;
	unsigned int frameNumber;
};

extern "C" void run_test_weighted_sum(
	dim3 const & grid_size,
	dim3 const & block_size,
	WeightedSumTestKernelParams const & params,
	const float * K_RESTRICT weights,			// [in]	 Features weights computed by the fitter kernel
	float * K_RESTRICT outputs,
	float * K_RESTRICT refs
);