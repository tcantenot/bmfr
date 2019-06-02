#pragma once

#include <vector>

#define FRAME_COUNT 60
#define SAVE_INTERMEDIARY_BUFFERS 0
#define ENABLE_DEBUG_OUTPUT_TMP_DATA 0
#define DEBUG_OUTPUT_FRAME_NUMBER 0

#define _CRT_SECURE_NO_WARNINGS
#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

// TODO detect FRAME_COUNT from the input files
//#define FRAME_COUNT 1
// Location where input frames and feature buffers are located
#define INPUT_DATA_PATH ../data/living-room/inputs
#define INPUT_DATA_PATH_STR STR(INPUT_DATA_PATH)
// camera_matrices.h is expected to be in the same folder
#include STR(INPUT_DATA_PATH/camera_matrices.h)
// These names are appended with NN.exr, where NN is the frame number
#define NOISY_FILE_NAME INPUT_DATA_PATH_STR"/color"
#define NORMAL_FILE_NAME INPUT_DATA_PATH_STR"/shading_normal"
#define POSITION_FILE_NAME INPUT_DATA_PATH_STR"/world_position"
#define ALBEDO_FILE_NAME INPUT_DATA_PATH_STR"/albedo"
#define OUTPUT_FOLDER "outputs/"
#define TO_OUTPUTS_FOLDER(file) OUTPUT_FOLDER ## file
#define OUTPUT_FILE_NAME TO_OUTPUTS_FOLDER("output")

constexpr size_t BlockSize = 32;

inline constexpr size_t ComputeWorksetWidth(size_t w)
{
	return BlockSize * ((w + BlockSize - 1) / BlockSize);
}

inline constexpr size_t ComputeWorksetHeight(size_t h)
{
	return BlockSize * ((h + BlockSize - 1) / BlockSize);
}

inline constexpr size_t ComputeWorksetWidthWithMargin(size_t w)
{
	return ComputeWorksetWidth(w) + BlockSize;
}

inline constexpr size_t ComputeWorksetHeightWithMargin(size_t h)
{
	return ComputeWorksetHeight(h) + BlockSize;
}

struct BufferDesc
{
	size_t w;
	size_t h;
	size_t x_stride;
	size_t y_stride;
	size_t byte_size;
};

BufferDesc GetRGB32FWorksetBufferDesc(size_t w, size_t h);

inline BufferDesc GetRGB32FBufferDesc(size_t w, size_t h)
{
	BufferDesc desc;
	desc.w = w;
	desc.h = h;
	desc.x_stride  = 3 * sizeof(float);
	desc.y_stride  = desc.w * 3 * sizeof(float);
	desc.byte_size = desc.w * desc.h * 3 * sizeof(float);
	return desc;
}

inline BufferDesc GetRGB32FWorksetBufferDesc(size_t w, size_t h)
{
	BufferDesc desc;
	desc.w = ComputeWorksetWidth(w);
	desc.h = ComputeWorksetHeight(h);
	desc.x_stride  = 3 * sizeof(float);
	desc.y_stride  = desc.w * desc.x_stride;
	desc.byte_size = desc.h * desc.y_stride;
	return desc;
}

inline BufferDesc GetRGB32FWorksetWithMarginBufferDesc(size_t w, size_t h)
{
	BufferDesc desc;
	desc.w = ComputeWorksetWidthWithMargin(w);
	desc.h = ComputeWorksetHeightWithMargin(h);
	desc.x_stride  = 3 * sizeof(float);
	desc.y_stride  = desc.w * desc.x_stride;
	desc.byte_size = desc.h * desc.y_stride;
	return desc;
}

#if 1

inline BufferDesc GetAlbedoBufferDesc(size_t w, size_t h) { return GetRGB32FBufferDesc(w, h); }
inline BufferDesc GetNormalsBufferDesc(size_t w, size_t h) { return GetRGB32FBufferDesc(w, h); }
inline BufferDesc GetPositionsBufferDesc(size_t w, size_t h) { return GetRGB32FBufferDesc(w, h); }
inline BufferDesc GetNoisy1sppBufferDesc(size_t w, size_t h) { return GetRGB32FBufferDesc(w, h); }
inline BufferDesc GetNoiseFree1sppBufferDesc(size_t w, size_t h) { return GetRGB32FBufferDesc(w, h); }
inline BufferDesc GetNoiseFree1sppAccumulatedBufferDesc(size_t w, size_t h) { return GetRGB32FBufferDesc(w, h); }
inline BufferDesc GetResultBufferDesc(size_t w, size_t h) { return GetRGB32FBufferDesc(w, h); }
inline BufferDesc GetPrevFramePixelCoordsBufferDesc(size_t w, size_t h)
{
	BufferDesc desc;
	desc.w = w;
	desc.h = h;
	desc.x_stride  = 2 * sizeof(float);
	desc.y_stride  = desc.w * desc.x_stride;
	desc.byte_size = desc.h * desc.y_stride;
	return desc;
}
inline BufferDesc GetPrevFrameBilinearSamplesValidityMaskBufferDesc(size_t w, size_t h)
{
	BufferDesc desc;
	desc.w = w;
	desc.h = h;
	desc.x_stride  = 2 * sizeof(unsigned char);
	desc.y_stride  = desc.w * desc.x_stride;
	desc.byte_size = desc.h * desc.y_stride;
	return desc;
}
inline BufferDesc GetNoiseFree1sppAccTonemappedBufferDesc(size_t w, size_t h) { return GetRGB32FBufferDesc(w, h); }
inline BufferDesc GetSppBufferDesc(size_t w, size_t h)
{
	BufferDesc desc;
	desc.w = w;
	desc.h = h;
	desc.x_stride  = sizeof(char);
	desc.y_stride  = desc.w * desc.x_stride;
	desc.byte_size = desc.h * desc.y_stride;
	return desc;
}

#else

inline BufferDesc GetAlbedoBufferDesc(size_t w, size_t h) { return GetRGB32FBufferDesc(w, h); }
inline BufferDesc GetNormalsBufferDesc(size_t w, size_t h) { return GetRGB32FWorksetBufferDesc(w, h); }
inline BufferDesc GetPositionsBufferDesc(size_t w, size_t h) { return GetRGB32FWorksetBufferDesc(w, h); }
inline BufferDesc GetNoisy1sppBufferDesc(size_t w, size_t h) { return GetRGB32FWorksetBufferDesc(w, h); }
inline BufferDesc GetNoiseFree1sppBufferDesc(size_t w, size_t h) { return GetRGB32FWorksetBufferDesc(w, h); }
inline BufferDesc GetNoiseFree1sppAccumulatedBufferDesc(size_t w, size_t h) { return GetRGB32FWorksetWithMarginBufferDesc(w, h); }
inline BufferDesc GetResultBufferDesc(size_t w, size_t h) { return GetRGB32FWorksetBufferDesc(w, h); }
inline BufferDesc GetPrevFramePixelCoordsBufferDesc(size_t w, size_t h)
{
	BufferDesc desc;
	desc.w = ComputeWorksetWidth(w);
	desc.h = ComputeWorksetHeight(h);
	desc.x_stride  = 2 * sizeof(float);
	desc.y_stride  = desc.w * desc.x_stride;
	desc.byte_size = desc.h * desc.y_stride;
	return desc;
}
inline BufferDesc GetPrevFrameBilinearSamplesValidityMaskBufferDesc(size_t w, size_t h)
{
	BufferDesc desc;
	desc.w = ComputeWorksetWidth(w);
	desc.h = ComputeWorksetHeight(h);
	desc.x_stride  = 2 * sizeof(unsigned char);
	desc.y_stride  = desc.w * desc.x_stride;
	desc.byte_size = desc.h * desc.y_stride;
	return desc;
}
inline BufferDesc GetNoiseFree1sppAccTonemappedBufferDesc(size_t w, size_t h) { return GetRGB32FWorksetBufferDesc(w, h); }
inline BufferDesc GetSppBufferDesc(size_t w, size_t h)
{
	BufferDesc desc;
	desc.w = ComputeWorksetWidth(w);
	desc.h = ComputeWorksetHeight(h);
	desc.x_stride  = sizeof(char);
	desc.y_stride  = desc.w * desc.x_stride;
	desc.byte_size = desc.h * desc.y_stride;
	return desc;
}

#endif

struct TmpData
{
	std::vector<float> normals;
	std::vector<float> positions;
	std::vector<float> noisy_1spp;
	std::vector<float> prev_frame_pixel_coords_buffer;
	std::vector<unsigned char> prev_frame_bilinear_samples_validity_mask;
	std::vector<unsigned char> spp;
	std::vector<float> features_buffer0;
	std::vector<float> features_buffer1;
	std::vector<float> features_weights_buffer;
	std::vector<float> features_min_max_buffer;
	std::vector<float> noisefree_1spp;
	std::vector<float> noisefree_1spp_accumulated;
	std::vector<float> noisefree_1spp_acc_tonemapped;
	std::vector<float> result;

};

