#pragma once

#include "bmfr.cuh"

// TODO:
struct BMFRFrameData
{
	float * world_normals;
	float * world_positions;
	float * noisy_1spp;
	//...
};


// Rescale features ////////////////////////////////////////////////////////////

extern "C" void run_rescale_features(
	dim3 const & grid_size,
	dim3 const & block_size,
	float * features,
	unsigned int n
);

// Accumulate noisy 1spp color kernel //////////////////////////////////////////

extern "C" void run_accumulate_noisy_data_frame0(
	dim3 const & grid_size,
	dim3 const & block_size,
	AccumulateNoisyDataKernelParams params,
	const float * K_RESTRICT frame_normals,				// [in]  Frame (world) normals
	const float * K_RESTRICT frame_positions,			// [in]  Frame world positions
	const float * K_RESTRICT frame_noisy_1spp,			// [in]  Frame noisy 1spp color buffer
		  float * K_RESTRICT frame_acc_noisy,			// [out] Accumulated noisy color
		  unsigned char * K_RESTRICT frame_acc_num_spp,	// [out] Accumulated number of samples (for CMA)
	#if USE_HALF_PRECISION_IN_FEATURES_DATA
	half * K_RESTRICT features_data						// [out] Features buffer (half-precision)
	#else
	float * K_RESTRICT features_data					// [out] Features buffer (single-precision)
	#endif
);

extern "C" void run_new_accumulate_noisy_data(
	dim3 const & grid_size,
	dim3 const & block_size,
	AccumulateNoisyDataKernelParams const & params,
	vec2 * K_RESTRICT out_prev_frame_pixel,					// [out] Previous frame pixel coordinates (after reprojection)
	unsigned char* K_RESTRICT accept_bools,					// [out] Validity mask of bilinear samples in previous frame (after reprojection)
	const float * K_RESTRICT frame_normals,					// [in]  Current  (world) normals
	const float * K_RESTRICT prev_frame_normals,			// [in]  Previous (world) normals
	const float * K_RESTRICT frame_positions,				// [in]  Current  world positions
	const float * K_RESTRICT prev_frame_positions,			// [in]  Previous world positions
	const float * K_RESTRICT frame_noisy_1spp,				// [in]  Frame noisy 1spp color buffer
		  float * K_RESTRICT frame_acc_noisy,				// [out] Current  noisy 1spp color
	const float * K_RESTRICT prev_frame_acc_noisy,			// [in]  Previous noisy 1spp color
	const unsigned char * K_RESTRICT prev_frame_acc_spp,	// [in]  Previous number of samples accumulated (for CMA)
		  unsigned char * K_RESTRICT frame_acc_num_spp,		// [out] Current  number of samples accumulated (for CMA)
	#if USE_HALF_PRECISION_IN_FEATURES_DATA
	half * K_RESTRICT features_data,						// [out] Features buffer (half-precision)
	#else
	float * K_RESTRICT features_data,						// [out] Features buffer (single-precision)
	#endif
	const mat4x4 prev_frame_camera_matrix,					// [in]  ViewProj matrix of previous frame
	const vec2 pixel_offset
);

// Fitter kernel ///////////////////////////////////////////////////////////////

extern "C" void run_new_fitter(
	dim3 const & grid_size,
	dim3 const & block_size,
	FitterKernelParams const & params,
	float * K_RESTRICT weights,					// [out] Features weights
	#if USE_HALF_PRECISION_IN_FEATURES_DATA
	half * K_RESTRICT features_buffer			// [out] Features buffer (half-precision)
	#else
	float * K_RESTRICT features_buffer			// [out] Features buffer (single-precision)
	#endif
);

extern "C" void run_fitter16bits(
	dim3 const & grid_size,
	dim3 const & block_size,
	FitterKernelParams params,
	//half * K_RESTRICT weights,			// [out] Features weights
	float * K_RESTRICT weights,			// [out] Features weights
	half * K_RESTRICT features_buffer	// [in]  Features buffer
);

// Weighted sum kernel /////////////////////////////////////////////////////////
// -> outputs the noise-free 1spp color estimate

extern "C" void run_new_weighted_sum(
	dim3 const & grid_size,
	dim3 const & block_size,
	WeightedSumKernelParams const & params,
	const float * K_RESTRICT weights,			// [in]	 Features weights computed by the fitter kernel
		  float * K_RESTRICT output,			// [out] Noise-free color estimate
	const float * K_RESTRICT current_normals,	// [in]  Current (world) normals
	const float * K_RESTRICT current_positions	// [in]  Current world positions
);


// Accumulate filtered data kernel /////////////////////////////////////////////
// -> outputs the noise-free accumulated color estimate + a tonemapped version w/ albedo

struct AccumulateFilteredDataKernelParams2
{
	unsigned int sizeX;
	unsigned int sizeY;
};

extern "C" void run_accumulate_filtered_data_frame0(
	dim3 const & grid_size,
	dim3 const & block_size,
	AccumulateFilteredDataKernelParams2 const & params,
	const float * K_RESTRICT filtered_frame,			// [in]  Noise free color estimate (computed as the weighted sum of the features)
	const float * K_RESTRICT albedo_buffer,				// [in]  Albedo buffer of the current frame (non-noisy)
		  float * K_RESTRICT tone_mapped_frame,			// [out] Accumulated and tonemapped noise-free color estimate
		  float * K_RESTRICT accumulated_frame			// [out] Current frame noise-free accumulated color estimate
);

extern "C" void run_new_accumulate_filtered_data(
	dim3 const & grid_size,
	dim3 const & block_size,
	AccumulateFilteredDataKernelParams2 const & params,
	const float * K_RESTRICT filtered_frame,			// [in]  Noise free color estimate (computed as the weighted sum of the features)
	const vec2 * K_RESTRICT in_prev_frame_pixel,		// [in]  Previous frame pixel coordinates (after reprojection)
	const unsigned char * K_RESTRICT accept_bools,		// [in]  Validity mask of bilinear samples in previous frame (after reprojection)
	const float * K_RESTRICT albedo_buffer,				// [in]  Albedo buffer of the current frame (non-noisy)
		  float * K_RESTRICT tone_mapped_frame,			// [out] Accumulated and tonemapped noise-free color estimate
	const unsigned char* K_RESTRICT current_spp,		// [in]	 Current number of samples accumulated (for CMA)
	const float * K_RESTRICT accumulated_prev_frame,	// [in]  Previous frame noise-free accumulated color estimate 
		  float * K_RESTRICT accumulated_frame			// [out] Current frame noise-free accumulated color estimate
);

// TAA kernel //////////////////////////////////////////////////////////////////

extern "C" void run_taa_frame0(
	dim3 const & grid_size,
	dim3 const & block_size,
	TAAKernelParams const & params,
	const float * K_RESTRICT new_frame,				// [in]	 Current frame color buffer
		  float * K_RESTRICT result_frame			// [out] Antialiased frame color buffer
);

extern "C" void run_new_taa(
	dim3 const & grid_size,
	dim3 const & block_size,
	TAAKernelParams const & params,
	const vec2 * K_RESTRICT in_prev_frame_pixel,	// [in]  Previous frame pixel coordinates (after reprojection)
	const float * K_RESTRICT new_frame,				// [in]	 Current frame color buffer
		  float * K_RESTRICT result_frame,			// [out] Antialiased frame color buffer
	const float * K_RESTRICT prev_frame				// [in]  Previous frame color buffer
);