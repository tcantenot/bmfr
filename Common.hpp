#pragma once

#include <vector>

#define ENABLE_DEBUG_OUTPUT_TMP_DATA 1
#define DEBUG_OUTPUT_FRAME_NUMBER 0

struct TmpData
{
	std::vector<float> normals;
	std::vector<float> positions;
	std::vector<float> noisy_1spp;
	std::vector<float> features_buffer;
	std::vector<unsigned char> spp; 
	std::vector<float> features_weights_buffer;
	std::vector<float> features_min_max_buffer;
};

