#pragma once

#include <vector>

#define FRAME_COUNT 2
#define ENABLE_DEBUG_OUTPUT_TMP_DATA 1
#define DEBUG_OUTPUT_FRAME_NUMBER 1

struct TmpData
{
	std::vector<float> normals;
	std::vector<float> positions;
	std::vector<float> noisy_1spp;
	std::vector<float> prev_frame_pixel_coords_buffer;
	std::vector<unsigned char> prev_frame_bilinear_samples_validity_mask;
	std::vector<unsigned char> spp;
	std::vector<float> features_buffer;
	std::vector<float> features_weights_buffer;
	std::vector<float> features_min_max_buffer;
	std::vector<float> noisefree_1spp;
	std::vector<float> noisefree_1spp_accumulated;
	std::vector<float> noisefree_1spp_acc_tonemapped;
	std::vector<float> result;

};

