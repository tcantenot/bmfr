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
#define OUTPUT_FOLDER "../outputs/"
#define TO_OUTPUTS_FOLDER(file) OUTPUT_FOLDER ## file
#define OUTPUT_FILE_NAME TO_OUTPUTS_FOLDER("output")

// TODO: turn size and dependent constants into variables (that will be baked as constant inside the kernel)

// TODO detect IMAGE_SIZES automatically from the input files
#define IMAGE_WIDTH 1280
#define IMAGE_HEIGHT 720


// ### Edit these defines if you want to experiment different parameters ###
// The amount of noise added to feature buffers to cancel sigularities
#define NOISE_AMOUNT 1e-2f

// The amount of new frame used in accumulated frame (1.f would mean no accumulation).
#define BLEND_ALPHA 0.2f
#define SECOND_BLEND_ALPHA 0.1f
#define TAA_BLEND_ALPHA 0.2f

// NOTE: if you want to use other than normal and world_position data you have to make
// it available in the first accumulation kernel and in the weighted sum kernel
#define NOT_SCALED_FEATURE_BUFFERS_STR \
"1.f,"\
"normal.x,"\
"normal.y,"\
"normal.z"

#define USE_SCALED_FEATURES 1

#if USE_SCALED_FEATURES
// The next features are not in the range from -1 to 1 so they are scaled to be from 0 to 1.
#define SCALED_FEATURE_BUFFERS_STR \
",world_position.x,"\
"world_position.y,"\
"world_position.z,"\
"world_position.x*world_position.x,"\
"world_position.y*world_position.y,"\
"world_position.z*world_position.z"
#else
#define SCALED_FEATURE_BUFFERS_STR ""
#endif

// ### Edit these defines to change optimizations for your target hardware ###
// If 1 uses ~half local memory space for R, but computing indexes is more complicated
#define COMPRESSED_R 1

// If 1 stores tmp_data to private memory when it is loaded for dot product calculation
#define CACHE_TMP_DATA 1

// If 1 features_data buffer is in half precision for faster load and store.
// NOTE: if world position values are greater than 256 this cannot be used because
// 256*256 is infinity in half-precision
#define USE_HALF_PRECISION_IN_FEATURES_DATA 0

// If 1 adds __attribute__((reqd_work_group_size(256, 1, 1))) to fitter and
// accumulate_noisy_data kernels. With some codes, attribute made the kernels faster and
// with some it slowed them down.
#define ADD_REQD_WG_SIZE 1

// These local sizes are used with 2D kernels which do not require spesific local size
// (Global sizes are always a multiple of 32)
#define LOCAL_WIDTH 8
#define LOCAL_HEIGHT 8
// Fastest on AMD Radeon Vega Frontier Edition was (LOCAL_WIDTH = 256, LOCAL_HEIGHT = 1)
// Fastest on Nvidia Titan Xp was (LOCAL_WIDTH = 32, LOCAL_HEIGHT = 1)


// ### Do not edit defines after this line unless you know what you are doing ###
// For example, other than 32x32 blocks are not supported
#define BLOCK_EDGE_LENGTH 32

#define BLOCK_PIXELS (BLOCK_EDGE_LENGTH * BLOCK_EDGE_LENGTH)

// Rounds image sizes up to next multiple of BLOCK_EDGE_LENGTH
#define WORKSET_WIDTH  (BLOCK_EDGE_LENGTH * ((IMAGE_WIDTH  + BLOCK_EDGE_LENGTH - 1) / BLOCK_EDGE_LENGTH))
#define WORKSET_HEIGHT (BLOCK_EDGE_LENGTH * ((IMAGE_HEIGHT + BLOCK_EDGE_LENGTH - 1) / BLOCK_EDGE_LENGTH))

// TODO: not sure that the buffers must be OUTPUT_SIZE: IMAGE_WITDH * IMAGE_HEIGHT should be enough
#define OUTPUT_SIZE (WORKSET_WIDTH * WORKSET_HEIGHT)

// 256 is the maximum local size on AMD GCN
// Synchronization within 32x32=1024 block requires unrolling four times
#define LOCAL_SIZE 256

// We need a margin of one block (BLOCK_EDGE_LENGTH) because we are offsetting the blocks
// at most one block to the left or the right to avoid blocky artifacts.
// So most of the time, we have 2 partially filled blocks and BLOCK_COUNT_X-1 full blocks on one row

// Example: image of 8x12 with blocks of 4x1

// No offset: 2x3 = 6 blocks
// |....|....|
// |....|....|
// |....|....|

// With some offset (here (-2, 0)): 3x3 = 9 blocks
// |--..|....|..--|
// |--..|....|..--|
// |--..|....|..--|

// Workset with margin width
#define WORKSET_WITH_MARGINS_WIDTH (WORKSET_WIDTH + BLOCK_EDGE_LENGTH)

// Workset with margin height
#define WORKSET_WITH_MARGINS_HEIGHT (WORKSET_HEIGHT + BLOCK_EDGE_LENGTH)

// Number of blocks in X dim in the workset with margin
#define WORKSET_WITH_MARGIN_BLOCK_COUNT_X (WORKSET_WITH_MARGINS_WIDTH  / BLOCK_EDGE_LENGTH)

// Number of blocks in Y dim in the workset with margin
#define WORKSET_WITH_MARGIN_BLOCK_COUNT_Y (WORKSET_WITH_MARGINS_HEIGHT / BLOCK_EDGE_LENGTH)

// Number of block in the workset with margin
#define WORKSET_WITH_MARGIN_BLOCK_COUNT (WORKSET_WITH_MARGIN_BLOCK_COUNT_X * WORKSET_WITH_MARGIN_BLOCK_COUNT_Y)

// Fitter kernel global range
#define FITTER_KERNEL_GLOBAL_RANGE (LOCAL_SIZE * WORKSET_WITH_MARGIN_BLOCK_COUNT)


/////////////////////////////////////

#define LOG(...) printf(__VA_ARGS__)

#if 0
#define DEBUG_LOG(...) LOG(__VA_ARGS__)
#else
#define DEBUG_LOG(...) (void)0
#endif

/////////////////////////////////////


constexpr size_t BlockSize = BLOCK_EDGE_LENGTH;

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

