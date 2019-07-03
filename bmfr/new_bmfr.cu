#include "new_bmfr.cuh"
#include "math.cuh"
#include "reduction.cuh"

// TODO
// - Try to pass everything as half (inputs and outputs)
//   --> #define WorldPositionType|WorldNormals|... float or half
// - LoadWorldPositions, LoadWorldNormals, ...

#define NORMALIZE_FEATURES_USING_4_BLOCKS 0
#define USE_FEATURES_VGPR_CACHE USE_HALF_PRECISION_IN_FEATURES_DATA


inline __device__ float Add(float lhs, float rhs) { return lhs + rhs; }
inline __device__ float Sub(float lhs, float rhs) { return lhs - rhs; }
inline __device__ float Mul(float lhs, float rhs) { return lhs * rhs; }
inline __device__ float Div(float lhs, float rhs) { return lhs / rhs; }

template <typename T>
struct Converter;

template <>
struct Converter<float>
{
	static inline __device__ float Convert(half x)  { return __half2float(x); }
	static inline __device__ float Convert(float x) { return x; }
};

template <>
struct Converter<half>
{
	static inline __device__ half Convert(half x)  { return x; }
	static inline __device__ half Convert(float x) { return __float2half(x); }
};


template <typename In, typename Out>
struct LoadStoreHelper;

template <>
struct LoadStoreHelper<float, float>
{
	static inline __device__ vec3 load3(float const * K_RESTRICT buffer, unsigned int index)
	{
		#if OPTIMIZE_LOAD_STORE
		return (*reinterpret_cast<vec4 const *>(buffer + index * 3)).xyz();
		#else
		return vec3(buffer[index * 3 + 0], buffer[index * 3 + 1], buffer[index * 3 + 2]);
		#endif
	}

	static inline __device__ void store3(float * K_RESTRICT buffer, unsigned int index, vec3 const & value)
	{
		#if OPTIMIZE_LOAD_STORE
		*reinterpret_cast<vec3 *>(buffer + index * 3) = value;
		#else
		buffer[index * 3 + 0] = value.x;
		buffer[index * 3 + 1] = value.y;
		buffer[index * 3 + 2] = value.z;
		#endif
	}
};

template <>
struct LoadStoreHelper<half, half>
{
	static inline __device__ vec3h load3(half const * K_RESTRICT buffer, unsigned int index)
	{
		#if OPTIMIZE_LOAD_STORE
		return (*reinterpret_cast<vec4h const *>(buffer + index * 3)).xyz();
		#else
		return vec3h(buffer[index * 3 + 0], buffer[index * 3 + 1], buffer[index * 3 + 2]);
		#endif
	}

	static inline __device__ void store3(half * K_RESTRICT buffer, unsigned int index, vec3h const & value)
	{
		#if OPTIMIZE_LOAD_STORE
		*reinterpret_cast<vec3h *>(buffer + index * 3) = value;
		#else
		buffer[index * 3 + 0] = value.x;
		buffer[index * 3 + 1] = value.y;
		buffer[index * 3 + 2] = value.z;
		#endif
	}
};

template <>
struct LoadStoreHelper<float, half>
{
	static inline __device__ vec3h load3(float const * K_RESTRICT buffer, unsigned int index)
	{
		#if OPTIMIZE_LOAD_STORE
		vec4 const * v = reinterpret_cast<vec4 const *>(buffer + index * 3);
		return vec3h(__float2half(v->x), __float2half(v->y), __float2half(v->z));
		#else
		return vec3h(__float2half(buffer[index * 3 + 0]),__float2half(buffer[index * 3 + 1]), __float2half(buffer[index * 3 + 2]));
		#endif
	}

	static inline __device__ void store3(half * K_RESTRICT buffer, unsigned int index, vec3 const & value)
	{
		#if OPTIMIZE_LOAD_STORE
		vec3h * v = reinterpret_cast<vec3h *>(buffer + index * 3);
		v->x = __float2half(value.x);
		v->y = __float2half(value.y);
		v->z = __float2half(value.z);
		#else
		buffer[index * 3 + 0] = __float2half(value.x);
		buffer[index * 3 + 1] = __float2half(value.y);
		buffer[index * 3 + 2] = __float2half(value.z);
		#endif
	}
};

template <>
struct LoadStoreHelper<half, float>
{
	static inline __device__ vec3 load3(half const * K_RESTRICT buffer, unsigned int index)
	{
		#if OPTIMIZE_LOAD_STORE
		vec4h const * v = reinterpret_cast<vec4h const *>(buffer + index * 3);
		return vec3(__half2float(v->x), __half2float(v->y), __half2float(v->z));
		#else
		return vec3(__half2float(buffer[index * 3 + 0]),__half2float(buffer[index * 3 + 1]), __half2float(buffer[index * 3 + 2]));
		#endif
	}

	static inline __device__ void store3(float * K_RESTRICT buffer, unsigned int index, vec3h const & value)
	{
		#if OPTIMIZE_LOAD_STORE
		vec3 * v = reinterpret_cast<vec3 *>(buffer + index * 3);
		v->x = __half2float(value.x);
		v->y = __half2float(value.y);
		v->z = __half2float(value.z);
		#else
		buffer[index * 3 + 0] = __half2float(value.x);
		buffer[index * 3 + 1] = __half2float(value.y);
		buffer[index * 3 + 2] = __half2float(value.z);
		#endif
	}
};


template <typename Out, typename In>
inline __device__ tvec3<Out> load3(In const * K_RESTRICT buffer, unsigned int index)
{
	return LoadStoreHelper<In, Out>::load3(buffer, index);
}

template <typename Out, typename In>
inline __device__ void store3(Out * K_RESTRICT buffer, unsigned int index, tvec3<In> const & value)
{
	LoadStoreHelper<In, Out>::store3(buffer, index, value);
}

inline __device__ void store_feature(half * buffer, unsigned int index, half value)
{
	buffer[index] = value;
}


template <int FitterBlockSize>
inline __device__ ivec2 GetBlockOffset(unsigned int frameNumber)
{
	switch(FitterBlockSize)
	{
		case 16: return ivec2(-FitterBlockSize / 2) + BLOCK_OFFSETS_16[frameNumber % BLOCK_OFFSETS_COUNT];
		case 32: return ivec2(-FitterBlockSize / 2) + BLOCK_OFFSETS_32[frameNumber % BLOCK_OFFSETS_COUNT];
		case 64: return ivec2(-FitterBlockSize / 2) + BLOCK_OFFSETS_64[frameNumber % BLOCK_OFFSETS_COUNT];
		default: return ivec2(0);
	}
}


template <int FitterBlockSize>
inline __device__ ivec2 PixelCoordsToShiftedPixelCoords(
	ivec2 const & pixelCoords,
	unsigned int frameNumber
)
{
	return pixelCoords + GetBlockOffset<FitterBlockSize>(frameNumber); 
}

template <int FitterBlockSize>
inline __device__ ivec2 ShiftedPixelCoordsToPixelCoords(
	ivec2 const & pixelCoords,
	unsigned int frameNumber
)
{
	return pixelCoords - GetBlockOffset<FitterBlockSize>(frameNumber);
}

// Compute features ////////////////////////////////////////////////////////////

template <typename T, typename U, typename FeatureType>
inline __device__ void compute_features_without_color(
	tvec3<T> const & world_position,
	tvec3<U> const & normal,
	FeatureType * features
)
{
	features[0]  = Converter<FeatureType>::Convert(1.0f);
	features[1]  = Converter<FeatureType>::Convert(normal.x);
	features[2]  = Converter<FeatureType>::Convert(normal.y);
	features[3]  = Converter<FeatureType>::Convert(normal.z);
	features[4]  = Converter<FeatureType>::Convert(world_position.x);
	features[5]  = Converter<FeatureType>::Convert(world_position.y);
	features[6]  = Converter<FeatureType>::Convert(world_position.z);
	features[7]  = Converter<FeatureType>::Convert(Mul(world_position.x, world_position.x));
	features[8]  = Converter<FeatureType>::Convert(Mul(world_position.y, world_position.y));
	features[9]  = Converter<FeatureType>::Convert(Mul(world_position.z, world_position.z));
}

template <typename T, typename U, typename V, typename FeatureType>
inline __device__ void compute_features(
	tvec3<T> const & world_position,
	tvec3<U> const & normal,
	tvec3<V> const & noisy_1spp_color,
	FeatureType * features
)
{
	compute_features_without_color(world_position, normal, features);
	features[10] = Converter<FeatureType>::Convert(noisy_1spp_color.x);
	features[11] = Converter<FeatureType>::Convert(noisy_1spp_color.y);
	features[12] = Converter<FeatureType>::Convert(noisy_1spp_color.z);
}

// Rescale features ////////////////////////////////////////////////////////////

template <int FitterBlockSize>
__global__ void rescale_world_positions_pr(
	RescaleFeaturesParams params,
	float const * world_positions,
	float * normalized_world_positions
)
{
	__shared__ float lds[FitterBlockSize * FitterBlockSize];
	__shared__ float block_min;
	__shared__ float block_max;

	const ivec2 gtid = ivec2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

	const int w = params.sizeX;
	const int h = params.sizeY;

	#if !NORMALIZE_FEATURES_USING_4_BLOCKS
	const ivec2 pixel_without_mirror = PixelCoordsToShiftedPixelCoords<FitterBlockSize>(gtid, params.frameNumber);

	// Pixel coordinates in [0, w-1]x[0, h-1]
	const ivec2 pixel = mirror2(pixel_without_mirror, ivec2(w, h));

	// Linear pixel index in image in [0, w*h-1]
	const int linear_pixel = pixel.y * w + pixel.x;

	// Current frame world position
	const vec3 v = load3<float>(world_positions, linear_pixel);

	// Note: assume group of size 1024
	const unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;
		
	lds[tid] = v.x;
	SyncThreads();
	parallel_reduction_min_1024(&block_min, lds, tid);
	lds[tid] = v.x;
	SyncThreads();
	parallel_reduction_max_1024(&block_max, lds, tid);
	float scaledX = -Min(-scale(v.x, block_min, block_max), 0.0f); // Remove NaN

	lds[tid] = v.y;
	SyncThreads();
	parallel_reduction_min_1024(&block_min, lds, tid);
	lds[tid] = v.y;
	SyncThreads();
	parallel_reduction_max_1024(&block_max, lds, tid);
	float scaledY = -Min(-scale(v.y, block_min, block_max), 0.0f);

	lds[tid] = v.z;
	SyncThreads();
	parallel_reduction_min_1024(&block_min, lds, tid);
	lds[tid] = v.z;
	SyncThreads();
	parallel_reduction_max_1024(&block_max, lds, tid);
	float scaledZ = -Min(-scale(v.z, block_min, block_max), 0.0f);

	#else // 4-blocks (not really a win in quality...)
	const ivec2 offset = GetBlockOffset<FitterBlockSize>(frameNumber);

	ivec2 offsets[4] =
	{
		ivec2(offset.x - offset.x / 2, offset.y - offset.y / 2),
		ivec2(offset.x - offset.x / 2, offset.y + offset.y / 2),
		ivec2(offset.x + offset.x / 2, offset.y - offset.y / 2),
		ivec2(offset.x + offset.x / 2, offset.y + offset.y / 2)
	};

	const ivec2 pixel_without_mirror = gtid + offset;
	const ivec2 pixel = mirror2(pixel_without_mirror, ivec2(w, h));
	const int linear_pixel = pixel.y * w + pixel.x;
	const vec3 v = load3<float>(world_positions, linear_pixel);

	const ivec2 pixel_without_mirror_0 = gtid + offsets[0];
	const ivec2 pixel_0 = mirror2(pixel_without_mirror_0, ivec2(w, h));
	const int linear_pixel_0 = pixel_0.y * w + pixel_0.x;
	vec3 v_0 = load3<float>(world_positions, linear_pixel_0);

	vec3 vmin = v_0;
	vec3 vmax = v_0;
	for(unsigned int i = 1; i < 4; ++i)
	{
		const ivec2 pixel_without_mirror_i = gtid + offsets[i];
		const ivec2 pixel_i = mirror2(pixel_without_mirror_i, ivec2(w, h));
		const int linear_pixel_i = pixel_i.y * w + pixel_i.x;
		vec3 v_i = load3<float>(world_positions, linear_pixel_i);
		vmin = Min(vmin, v_i);
		vmax = Max(vmax, v_i);
	}
	
	// Note: assume group of size 1024
	const unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;
		
	lds[tid] = vmin.x;
	SyncThreads();
	parallel_reduction_min_1024(&block_min, lds, tid);
	lds[tid] = vmax.x;
	SyncThreads();
	parallel_reduction_max_1024(&block_max, lds, tid);
	float scaledX = -Min(-scale(v.x, block_min, block_max), 0.0f); // Remove NaN

	lds[tid] = vmin.y;
	SyncThreads();
	parallel_reduction_min_1024(&block_min, lds, tid);
	lds[tid] = vmax.y;
	SyncThreads();
	parallel_reduction_max_1024(&block_max, lds, tid);
	float scaledY = -Min(-scale(v.y, block_min, block_max), 0.0f);

	lds[tid] = vmin.z;
	SyncThreads();
	parallel_reduction_min_1024(&block_min, lds, tid);
	lds[tid] = vmax.z;
	SyncThreads();
	parallel_reduction_max_1024(&block_max, lds, tid);
	float scaledZ = -Min(-scale(v.z, block_min, block_max), 0.0f);
	#endif

	if(pixel_without_mirror.x >= 0 && pixel_without_mirror.x < w &&
	   pixel_without_mirror.y >= 0 && pixel_without_mirror.y < h
	)
	{
		store3(normalized_world_positions, linear_pixel, vec3(scaledX, scaledY, scaledZ));
	}
}

extern "C" void run_rescale_world_positions_pr(
	dim3 const & grid_size,
	dim3 const & block_size,
	RescaleFeaturesParams const & params,
	float const * world_positions,
	float * normalized_world_positions
)
{
	switch(params.fitterBlockSize)
	{
		case 16:rescale_world_positions_pr<16><<<grid_size, block_size>>>(params, world_positions,	normalized_world_positions); break;
		case 32: rescale_world_positions_pr<32><<<grid_size, block_size>>>(params, world_positions,	normalized_world_positions); break;
		case 64: rescale_world_positions_pr<64><<<grid_size, block_size>>>(params, world_positions,	normalized_world_positions); break;
		default: break;
	}
}

// Accumulate noisy 1spp color kernel //////////////////////////////////////////

template <int FitterBlockSize, typename NormalType, typename PosType, typename InColorType, typename OutColorType, typename FeaturesType>
__global__ void template_accumulate_noisy_data_frame0(
	AccumulateNoisyDataKernelParams params,
	const NormalType * K_RESTRICT frame_normals,			// [in]  Frame (world) normals
	const PosType * K_RESTRICT frame_normalized_positions,	// [in]  Frame normalized world positions
	const InColorType * K_RESTRICT frame_noisy_1spp,		// [in]  Frame noisy 1spp color buffer
	OutColorType * K_RESTRICT frame_acc_noisy,				// [out] Frame accumulated noisy color
	unsigned char * K_RESTRICT frame_acc_num_spp,			// [out] Frame accumulated number of samples (for CMA)
	FeaturesType * K_RESTRICT features_data					// [out] Features buffer
)
{
	const ivec2 gtid = ivec2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

	const int w = params.sizeX;
	const int h = params.sizeY;

	// Mirror indexed of the input. x and y are always less than one size out of bounds if image dimensions are bigger than block size
	const ivec2 pixel_without_mirror = PixelCoordsToShiftedPixelCoords<FitterBlockSize>(gtid, params.frameNumber);

	// Pixel coordinates in [0, w-1]x[0, h-1]
	const ivec2 pixel = mirror2(pixel_without_mirror, ivec2(w, h));

	// Linear pixel index in image in [0, w*h-1]
	const int linear_pixel = pixel.y * w + pixel.x;
   
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
	const tvec3<InColorType> current_color = load3<InColorType>(frame_noisy_1spp, linear_pixel);

	// Current frame normalized world position ([0, 1])
	const tvec3<PosType> normalized_world_position = load3<PosType>(frame_normalized_positions, linear_pixel);

	// Current frame (world) normal
	const tvec3<NormalType> normal = load3<NormalType>(frame_normals, linear_pixel);

	// Compute the set of feature buffers used in the fitting
	FeaturesType features[BUFFER_COUNT];
	compute_features(normalized_world_position, normal, current_color, features);

	const unsigned int x_block = gtid.x / FitterBlockSize; // Block coordinate x
	const unsigned int y_block = gtid.y / FitterBlockSize; // Block coordinate y
	const unsigned int x_in_block = gtid.x % FitterBlockSize; // Thread coordinate x inside block in [0, FitterBlockSize-1]
	const unsigned int y_in_block = gtid.y % FitterBlockSize; // Thread coordinate y inside block in [0, FitterBlockSize-1]

	const unsigned int numPixelsInBlock = FitterBlockSize * FitterBlockSize;
	const unsigned int features_base_offset = x_in_block + y_in_block * FitterBlockSize +
		x_block * numPixelsInBlock * BUFFER_COUNT +
		y_block * params.worksetWithMarginBlockCountX *
		numPixelsInBlock * BUFFER_COUNT;
	
	// TODO: change layout of features buffer to allow 128-bit loads?
	// --> | Block 0 thread 0 feature 0 | Block 0 thread 0 feature 1 | ... | Block 0 thread 1 feature 0 | ... | Block N thread 0 feature 0 | ... | Block N thread T feature M |
	for(unsigned int featureIndex = 0; featureIndex < BUFFER_COUNT; ++featureIndex)
	{
		// Offset in feature buffer (data are concatenated)
		// | Block 0 thread 0 feature 0 | Block 0 thread 1 feature 0 | ... | Block 0 thread 0 feature M | ... | Block 1 thread 0 feature 0 | ... | Block N thread 0 feature 0 | ... | Block N thread T feature M |
		const unsigned int featureOffset = features_base_offset + featureIndex * numPixelsInBlock;
		store_feature(features_data, featureOffset, features[featureIndex]);
	}

	// The kernel works on a workset of size WORKSET_WITH_MARGINS_WIDTH x WORKSET_WITH_MARGINS_HEIGHT
	// -> the extra block margin is used to handle the offsets applied to reduce the block artifacts.
	// [Section 3.5]: "To aid the reduction of blockiness, BMFR processes each frame over a grid of non-overlapping
	//	blocks which is displaced with random offsets. These offsets prevent the artifacts that would arise from reusing
	// same block positions on a static scene with a static camera."
	// --> Only the pixels inside the image (after applying the offsets) should write to the output data that
	// have the same size of the input image
	if(pixel_without_mirror.x >= 0 && pixel_without_mirror.x < w &&
	   pixel_without_mirror.y >= 0 && pixel_without_mirror.y < h
	)
	{
		store3(frame_acc_noisy, linear_pixel, current_color); // Accumulated noisy color
		frame_acc_num_spp[linear_pixel] = 1; // Store current number of samples accumulated (for CMA)
	}
}

extern "C" void run_accumulate_noisy_data_frame0(
	dim3 const & grid_size,
	dim3 const & block_size,
	AccumulateNoisyDataKernelParams params,
	const float * K_RESTRICT frame_normals,					// [in]  Frame (world) normals
	const float * K_RESTRICT frame_normalized_positions,	// [in]  Frame normalized world positions
	const float * K_RESTRICT frame_noisy_1spp,				// [in]  Frame noisy 1spp color buffer
		  float * K_RESTRICT frame_acc_noisy,				// [out] Accumulated noisy color
		  unsigned char * K_RESTRICT frame_acc_num_spp,		// [out] Accumulated number of samples (for CMA)
	#if USE_HALF_PRECISION_IN_FEATURES_DATA
	half * K_RESTRICT features_data							// [out] Features buffer (half-precision)
	#else
	float * K_RESTRICT features_data						// [out] Features buffer (single-precision)
	#endif
)
{
	switch(params.fitterBlockSize)
	{
		case 16:
		{
			template_accumulate_noisy_data_frame0<16><<<grid_size, block_size>>>(
				params,
				frame_normals,
				frame_normalized_positions,
				frame_noisy_1spp,
				frame_acc_noisy,
				frame_acc_num_spp,
				features_data
			);
			break;
		}

		case 32:
		{
			template_accumulate_noisy_data_frame0<32><<<grid_size, block_size>>>(
				params,
				frame_normals,
				frame_normalized_positions,
				frame_noisy_1spp,
				frame_acc_noisy,
				frame_acc_num_spp,
				features_data
			);
			break;
		}

		case 64:
		{
			template_accumulate_noisy_data_frame0<64><<<grid_size, block_size>>>(
				params,
				frame_normals,
				frame_normalized_positions,
				frame_noisy_1spp,
				frame_acc_noisy,
				frame_acc_num_spp,
				features_data
			);
			break;
		}

		default: break;
	}
}

template <int FitterBlockSize, typename NormalType, typename PosType, typename InColorType, typename OutColorType, typename FeaturesType>
__global__ void template_accumulate_noisy_data(
	AccumulateNoisyDataKernelParams params,
	vec2 * K_RESTRICT out_prev_frame_pixel,					// [out] Previous frame pixel coordinates (after reprojection)
	unsigned char* K_RESTRICT accept_bools,					// [out] Validity mask of bilinear samples in previous frame (after reprojection)
	const NormalType * K_RESTRICT frame_normals,			// [in]  Current  frame (world) normals
	const NormalType * K_RESTRICT prev_frame_normals,		// [in]  Previous frame (world) normals
	const PosType * K_RESTRICT frame_positions,				// [in]  Current  frame world positions
	const PosType * K_RESTRICT prev_frame_positions,		// [in]  Previous frame world positions
	const PosType * K_RESTRICT frame_normalized_positions,	// [in]  Frame normalized world positions
	const InColorType * K_RESTRICT frame_noisy_1spp,		// [in]  Frame noisy 1spp color
		  OutColorType * K_RESTRICT frame_acc_noisy,		// [out] Current  frame accumulated noisy color
	const OutColorType * K_RESTRICT prev_frame_acc_noisy,	// [in]  Previous frame accumulated noisy color
	const unsigned char * K_RESTRICT prev_frame_acc_spp,	// [in]  Previous frame accumulated number of samples (for CMA)
		  unsigned char * K_RESTRICT frame_acc_num_spp,		// [out] Current  frame accumulated number of samples (for CMA)
	FeaturesType * K_RESTRICT features_data,				// [out] Features buffer
	const mat4x4 prev_frame_camera_matrix,					// [in]  ViewProj matrix of previous frame
	const vec2 pixel_offset
)
{
	const ivec2 gtid = ivec2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

	const int w = params.sizeX;
	const int h = params.sizeY;

	// Mirror indexed of the input. x and y are always less than one size out of bounds if image dimensions are bigger than block size
	const ivec2 pixel_without_mirror = PixelCoordsToShiftedPixelCoords<FitterBlockSize>(gtid, params.frameNumber);

	// Pixel coordinates in [0, w-1]x[0, h-1]
	const ivec2 pixel = mirror2(pixel_without_mirror, ivec2(w, h));

	// Linear pixel index in image in [0, w*h-1]
	const int linear_pixel = pixel.y * w + pixel.x;
   
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
	const vec3 current_color = load3<float>(frame_noisy_1spp, linear_pixel);

	// Current frame normalized world position ([0, 1])
	const tvec3<PosType> normalized_world_position = load3<PosType>(frame_normalized_positions, linear_pixel);

	// Current frame (world) normal
	const vec3 normal = load3<float>(frame_normals, linear_pixel);

	// Project current world position into previous frame with the previous ViewProj matrix

	// Current frame world position
	// TODO: instead of comparing full world positions, use only depth of current, previous and
	// reprojected frame to detect (dis)occlusion
	const vec4 world_position = vec4(load3<float>(frame_positions, linear_pixel), 1.f);

	// Matrix multiplication and normalization to 0..1
	vec2 prev_frame_uv;

	// TODO: send matrix transposed
	prev_frame_uv.x = Dot(prev_frame_camera_matrix.row(0), world_position); // Transform x
	prev_frame_uv.y = Dot(prev_frame_camera_matrix.row(1), world_position); // Transform y
	// No need for z-buffer in accumulation of the noisy data
	// -> might be useful if we use it to detect disocclusion
	// --> compare previous z (store previous frame Z-buffer) with prev_frame_pixel_uv.z
	//prev_frame_uv.z = Dot(prev_frame_camera_matrix.row(2), world_position); // Transform z
	prev_frame_uv /= Dot(prev_frame_camera_matrix.row(3), world_position);

	prev_frame_uv = prev_frame_uv * vec2(0.5f) + vec2(0.5f);

	// Compute the pixel coordinates in the previous frame (in [0, w-1]x[0, h-1])
	vec2 prev_frame_pixel_f = prev_frame_uv * vec2(w, h);

	// Apply offset (TODO: what offset??? seems to always be 0.5... Maybe TAA/ray subpixel offsets -> send 1-pixel_offset.y)
	// TODO: try to remove this
	prev_frame_pixel_f -= vec2(pixel_offset.x, 1 - pixel_offset.y);

	// Convert into integer pixel coordinates (round down)
	const ivec2 prev_frame_pixel_i = FloatToIntRd(prev_frame_pixel_f);

	// Compute bilinear weights (for bilinear sampling)
	// TODO: implement bicubic Catmull-Rom (for sharpness)? => would need to perform more fetches and store more "validity bits" in mask
	const ivec2 offsets[4] = { ivec2(0, 0), ivec2(1, 0), ivec2(0, 1), ivec2(1, 1) };

	const vec2 prev_pixel_fract = prev_frame_pixel_f - vec2(prev_frame_pixel_i);
	const vec2 one_minus_prev_pixel_fract = 1.f - prev_pixel_fract;

	float weights[4];
	weights[0] = one_minus_prev_pixel_fract.x * one_minus_prev_pixel_fract.y;
	weights[1] = prev_pixel_fract.x           * one_minus_prev_pixel_fract.y;
	weights[2] = one_minus_prev_pixel_fract.x * prev_pixel_fract.y;
	weights[3] = prev_pixel_fract.x           * prev_pixel_fract.y;
	float total_weight = 0.f;

	// Bit mask telling which previous frame (bilinear) samples are valid under reprojection into current frame
	unsigned char store_accept = 0x00;
	vec3 previous_color = vec3(0.f, 0.f, 0.f);
	float sample_spp = 0.f;

	// Bilinear sampling
	for(int i = 0; i < 4; ++i)
	{
		ivec2 sample_location = prev_frame_pixel_i + offsets[i];

		// Check if previous frame color can be used based on its screen location
		if(sample_location.x >= 0 && sample_location.y >= 0 &&
			sample_location.x < w  && sample_location.y < h
		)
		{
			const int linear_sample_location = sample_location.y * w + sample_location.x;

			// Fetch previous frame world position
			vec3 prev_world_position = load3<float>(prev_frame_positions, linear_sample_location);

			// TODO: find a another metric to discard wrong history
			// -> world position is normalized to [0, 1]...
			// OR bind both normalized and non-normalized
			// Compute world distance squared
			vec3 position_difference = prev_world_position - world_position.xyz();
			float position_distance_squared = Dot(position_difference, position_difference);

			// World position distance discard
			if(position_distance_squared < float(POSITION_LIMIT_SQUARED))
			{
				// Fetch previous frame normal
				vec3 prev_normal = load3<float>(prev_frame_normals, linear_sample_location);

				// Distance of the normals
				// TODO: could use some other distance metric (e.g. angle), but we use hard
				// experimentally found threshold -> means that the metric doesn't matter.
				vec3 normal_difference = prev_normal - normal;
				float normal_distance_squared = Dot(normal_difference, normal_difference);

				if(normal_distance_squared < float(NORMAL_LIMIT_SQUARED))
				{
					// Pixel passes all tests so store it to "validity bitmask"
					store_accept |= 1 << i;

					// Accumulate number of samples
					sample_spp += weights[i] * float(prev_frame_acc_spp[linear_sample_location]);

					// Accumulate previous noisy 1spp color
					previous_color += weights[i] * load3<float>(prev_frame_acc_noisy, linear_sample_location);

					// Acumulate weights
					total_weight += weights[i];
				}
			}
		}
	}

	// Blending factor with history buffer
	// Blend_alpha 1.f means that only current frame color is used. The value is changed if sample from previous frame can be used
	float blend_alpha = 1.f;
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
		blend_alpha = Max(blend_alpha, BLEND_ALPHA);
	}

	// Store new spp
	unsigned char new_spp = 1;
	if(blend_alpha < 1.f) // alpha = 1.f means we ignore history
	{
		// Note: we accumulate at most 255 samples for the cumulative moving average (which is more than enough because of
		// the threshold BLEND_ALPHA that switch to exponential moving average).
		// E.g: BLEND_ALPHA = 0.2 = 1 / (n + 1) <=> n = (1 - 0.2) / 0.2 = 4 => above 4 samples for a pixel, we switch to
		// exponential moving average with alpha = 20%
		// max n = 255 <=> min BLEND_ALPHA = 1.0 / (255 + 1) = 0.0039
		
		// TODO: store "validity mask" along with the "spp" in 8 bits: 4-bit validity mask | 4-bit spp
		// 4-bit spp <=> max n = 2^4-1 = 15 <=> min BLEND_ALPHA = 1.0 / (5 + 1) = 0.0625

		new_spp = (sample_spp > 254.f) ? 255 : convert_uchar_sat_rte(sample_spp) + 1;
	}

	vec3 new_color = Lerp(previous_color, current_color, blend_alpha);

	// Compute the set of feature buffers used in the fitting
	FeaturesType features[BUFFER_COUNT];
	compute_features(normalized_world_position, normal, new_color, features);

	const unsigned int x_block = gtid.x / params.fitterBlockSize; // Block coordinate x
	const unsigned int y_block = gtid.y / params.fitterBlockSize; // Block coordinate y
	const unsigned int x_in_block = gtid.x % params.fitterBlockSize; // Thread coordinate x inside block in [0, blockSize-1]
	const unsigned int y_in_block = gtid.y % params.fitterBlockSize; // Thread coordinate y inside block in [0, blockSize-1]

	const unsigned int numPixelsInBlock = params.fitterBlockSize * params.fitterBlockSize;

	// TODO: send constants so that:
	// features_base_offset = x_in_block * params.featureOffsetScale0 + y_in_block * params.featureOffsetScale1;
	const unsigned int features_base_offset = x_in_block + y_in_block * params.fitterBlockSize +
		x_block * numPixelsInBlock * BUFFER_COUNT +
		y_block * params.worksetWithMarginBlockCountX *
		numPixelsInBlock * BUFFER_COUNT;
	
	// TODO: change layout of features buffer to allow 128-bit loads?
	// --> | Block 0 thread 0 feature 0 | Block 0 thread 0 feature 1 | ... | Block 0 thread 1 feature 0 | ... | Block N thread 0 feature 0 | ... | Block N thread T feature M |
	for(unsigned int featureIndex = 0; featureIndex < BUFFER_COUNT; ++featureIndex)
	{
		// Offset in feature buffer (data are concatenated)
		// | Block 0 thread 0 feature 0 | Block 0 thread 1 feature 0 | ... | Block 0 thread 0 feature M | ... | Block 1 thread 0 feature 0 | ... | Block N thread 0 feature 0 | ... | Block N thread T feature M |
		const unsigned int featureOffset = features_base_offset + featureIndex * numPixelsInBlock;
		store_feature(features_data, featureOffset, features[featureIndex]);
	}

	// The kernel works on a workset of size WORKSET_WITH_MARGINS_WIDTH x WORKSET_WITH_MARGINS_HEIGHT
	// -> the extra block margin is used to handle the offsets applied to reduce the block artifacts.
	// [Section 3.5]: "To aid the reduction of blockiness, BMFR processes each frame over a grid of non-overlapping
	//	blocks which is displaced with random offsets. These offsets prevent the artifacts that would arise from reusing
	// same block positions on a static scene with a static camera."
	// --> Only the pixels inside the image (after applying the offsets) should write to the output data that
	// have the same size of the input image
	if(pixel_without_mirror.x >= 0 && pixel_without_mirror.x < w &&
	   pixel_without_mirror.y >= 0 && pixel_without_mirror.y < h
	)
	{
		store3(frame_acc_noisy, linear_pixel, new_color); // Accumulated noisy 1spp
		out_prev_frame_pixel[linear_pixel] = prev_frame_pixel_f; // Previous frame pixel coordinates (to sample history)
		accept_bools[linear_pixel] = store_accept; // "Previous frame bilinear samples validity" bitmask
		frame_acc_num_spp[linear_pixel] = new_spp; // Store current number of samples accumulated (for CMA)

		// Kernel debug: stored in acc_noisy buffer
		#if 0
		vec3 debug = vec3(0.0f);
		//debug = vec3(prev_frame_uv.x, prev_frame_uv.y, 0);
		//debug = vec3(blend_alpha);
		debug = HeatMap(Saturate(float(new_spp) / 255.f));
		//debug = vec3(float(store_accept > 0));
		//debug = vec3(float(store_accept == ((1 << 4) - 1)));
		store3(acc_noisy, linear_pixel, debug);
		#endif
	}
}


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
	const float * K_RESTRICT frame_normalized_positions,	// [in]  Frame normalized world positions
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
)
{
	switch(params.fitterBlockSize)
	{
		case 16:
		{
			template_accumulate_noisy_data<16><<<grid_size, block_size>>>(
				params,
				out_prev_frame_pixel,
				accept_bools,
				frame_normals,
				prev_frame_normals,
				frame_positions,
				prev_frame_positions,
				frame_normalized_positions,
				frame_noisy_1spp,
				frame_acc_noisy,
				prev_frame_acc_noisy,
				prev_frame_acc_spp,
				frame_acc_num_spp,
				features_data,
				prev_frame_camera_matrix,
				pixel_offset
			);
			break;
		}

		case 32:
		{
			template_accumulate_noisy_data<32><<<grid_size, block_size>>>(
				params,
				out_prev_frame_pixel,
				accept_bools,
				frame_normals,
				prev_frame_normals,
				frame_positions,
				prev_frame_positions,
				frame_normalized_positions,
				frame_noisy_1spp,
				frame_acc_noisy,
				prev_frame_acc_noisy,
				prev_frame_acc_spp,
				frame_acc_num_spp,
				features_data,
				prev_frame_camera_matrix,
				pixel_offset
			);
			break;
		}

		case 64:
		{
			template_accumulate_noisy_data<64><<<grid_size, block_size>>>(
				params,
				out_prev_frame_pixel,
				accept_bools,
				frame_normals,
				prev_frame_normals,
				frame_positions,
				prev_frame_positions,
				frame_normalized_positions,
				frame_noisy_1spp,
				frame_acc_noisy,
				prev_frame_acc_noisy,
				prev_frame_acc_spp,
				frame_acc_num_spp,
				features_data,
				prev_frame_camera_matrix,
				pixel_offset
			);
			break;
		}

		default: break;
	}
}

// Fitter kernel ///////////////////////////////////////////////////////////////

template <int FitterBlockSize, unsigned int LocalSize>
__global__ void template_fitter(
	FitterKernelParams params,
	float * K_RESTRICT weights,					// [out] Features weights
	#if USE_HALF_PRECISION_IN_FEATURES_DATA
	half * K_RESTRICT features_buffer			// [out] Features buffer (half-precision)
	#else
	float * K_RESTRICT features_buffer			// [out] Features buffer (single-precision)
	#endif
)
{
	static_assert(FitterBlockSize % LocalSize == 0, "FitterBlockSize must be divisible by LocalSize");

	// TODO: send as define for cpp side
	#if COMPRESSED_R
    //const auto r_size = ((buffer_count - 2) * (buffer_count - 1) / 2) * sizeof(cl_float3);
	#define R_SHARED_DATA_SIZE ((BUFFER_COUNT - 2) * (BUFFER_COUNT - 1) / 2)
	#else
    //const auto r_size = (buffer_count - 2) * (buffer_count - 2) * sizeof(cl_float3);
	#define R_SHARED_DATA_SIZE ((BUFFER_COUNT - 2) * (BUFFER_COUNT - 2))
	#endif

	__shared__ float pr_shared_data[LocalSize];			// Shared memory used to perform parallel reduction
	__shared__ float u_vec_sdata[FitterBlockSize];		// Shared memory used to store the 'u' vectors
	__shared__ cvec3 r_mat_sdata[R_SHARED_DATA_SIZE];	// Shared memory used to store the R matrices of the QR factorization (vec3 -> one per color channel)
	__shared__ float u_length_squared;					// Shared memory variable that holds the 'u' vector square length
	__shared__ float dotProd;							// Shared memory variable that holds the dot product of...
	__shared__ float vec_length;						// Shared memory variable that holds the vec length			

	float * pSharedData = &pr_shared_data[0];
	float * u_vec = &u_vec_sdata[0];
	cvec3 * r_mat = &r_mat_sdata[0];

	const int groupId = blockIdx.x;
	const int threadId = threadIdx.x; // in [0, 255]

	const unsigned int blockIndexX = groupId % params.worksetWithMarginBlockCountX;
	const unsigned int blockIndexY = groupId / params.worksetWithMarginBlockCountX;
	const unsigned int linearBlockIndex = blockIndexY * params.worksetWithMarginBlockCountX + blockIndexX;
	const unsigned int threadFeaturesBuffersOffset = linearBlockIndex * BUFFER_COUNT * FitterBlockSize + threadId;

	const unsigned int baseSeed = params.frameNumber * BUFFER_COUNT * FitterBlockSize + threadId;
	
	// Non square matrices require processing every column.
	// Otherwise result is OKish, but R is not upper triangular matrix
	const int limit = (BUFFER_COUNT == FitterBlockSize) ? BUFFER_COUNT - 1 : BUFFER_COUNT;

	
	#if USE_FEATURES_VGPR_CACHE
	const unsigned int FeaturesCacheSize = (FitterBlockSize / LocalSize) * BUFFER_COUNT;
	#if USE_HALF_PRECISION_IN_FEATURES_DATA
	half featuresCache[FeaturesCacheSize];
	#else
	float featuresCache[FeaturesCacheSize];
	#endif

	for(unsigned int featureIndex = 0; featureIndex < BUFFER_COUNT; ++featureIndex)
	{
		const unsigned int baseFeatureOffset = featureIndex * FitterBlockSize + threadFeaturesBuffersOffset;
		const unsigned int baseFeaturesCacheOffset = featureIndex * (FitterBlockSize / LocalSize);

		for(int subVector = 0; subVector < (FitterBlockSize / LocalSize); ++subVector)
		{
			const unsigned int featureOffset = subVector * LocalSize + baseFeatureOffset;
			const unsigned int featuresCacheOffset = baseFeaturesCacheOffset + subVector;
			#if USE_HALF_PRECISION_IN_FEATURES_DATA
			featuresCache[featuresCacheOffset] = features_buffer[featureOffset];
			#else
			featuresCache[featuresCacheOffset] = load_feature(features_buffer, featureOffset);
			#endif
		}
	}
	#endif // USE_FEATURES_VGPR_CACHE

	// Compute R
	for(int col = 0; col < limit; col++)
	{
		// Note: the last 3 features values are the 3 channels of the color (not used for the regression)
		int col_limited = Min(col, BUFFER_COUNT - 3);

		// Load new column into memory
		const int featureIndex = col;

		#if USE_FEATURES_VGPR_CACHE
		const unsigned int baseFeaturesCacheOffset = featureIndex * (FitterBlockSize / LocalSize);
		#else
		const unsigned int baseFeatureOffset = featureIndex * FitterBlockSize + threadFeaturesBuffersOffset;
		#endif

		float tmp_sum_value = 0.f;

		// Manual unrolling for parallel reduction in case FitterBlockSize is greater than LocalSize
		// as the reduction operates on LocalSize elements.
		for(int subVector = 0; subVector < (FitterBlockSize / LocalSize); ++subVector)
		{
			// Load feature
			#if USE_FEATURES_VGPR_CACHE
				const unsigned int featuresCacheOffset = baseFeaturesCacheOffset + subVector;
				#if USE_HALF_PRECISION_IN_FEATURES_DATA
				float tmp = HalfToFloat(featuresCache[featuresCacheOffset]);
				#else
				float tmp = featuresCache[featuresCacheOffset];
				#endif
			#else
				const unsigned int featureOffset = subVector * LocalSize + baseFeatureOffset;
				float tmp = load_feature(features_buffer, featureOffset);
			#endif

			// Store the feature in shared memory
			const int index = subVector * LocalSize + threadId;
			u_vec[index] = tmp;

			if(index >= col_limited + 1)
			{
				tmp_sum_value += tmp * tmp;
			}
		}
		SyncThreads();

		// Find length of vector in A's column with reduction sum function
		pSharedData[threadId] = tmp_sum_value;
		SyncThreads();
		parallel_reduction_sum<LocalSize>(&vec_length, pSharedData, threadId);

		// NOTE: GCN Opencl compiler can do some optimization with this because if
		// initially wanted col_limited is used to select wich work-item runs which branch
		// it is slower. However using col produces the same result.
		float r_value;
		if(threadId < col)
		{
			// Copy u_vec value
			r_value = u_vec[threadId];
		}
		else if(threadId == col)
		{
			u_length_squared = vec_length;
			vec_length = Sqrt(vec_length + u_vec[col_limited] * u_vec[col_limited]);
			u_vec[col_limited] -= vec_length;
			u_length_squared += u_vec[col_limited] * u_vec[col_limited];

			// (u_length_squared is now updated length squared)
			r_value = vec_length;
		}
		else if(threadId > col) //Could have "&& threadId <  R_EDGE" but this is little bit faster
		{
			// Last values on every column are zeros
			r_value = 0.0f;
		}

		int id_limited = Min(threadId, BUFFER_COUNT - 3);
		if(col < BUFFER_COUNT - 3)
			store_r_mat_broadcast(r_mat, col_limited, id_limited, r_value);
		else
			store_r_mat_channel(r_mat, col_limited, id_limited, col - BUFFER_COUNT + 3, r_value);
		SyncThreads();

		// Transform further columns of A
		// NOTE: three last columns are three color channels of noisy data. However,
		// they all need to be transfomed as they were column indexed (buffers - 3)
		for(int featureIndex = col_limited+1; featureIndex < BUFFER_COUNT; ++featureIndex)
		{
			#if USE_FEATURES_VGPR_CACHE
			const unsigned int baseFeaturesCacheOffset = featureIndex * (FitterBlockSize / LocalSize);
			#else
			const unsigned int baseFeatureOffset = featureIndex * FitterBlockSize + threadFeaturesBuffersOffset;
			#endif

			const unsigned int baseFeatureSeed = featureIndex * FitterBlockSize + baseSeed;

			// Starts by computing dot product with reduction sum function
			#if CACHE_TMP_DATA
			// No need to load features_buffer twice because each work-item first copies value for
			// dot product computation and then modifies the same value
			float tmp_data_private_cache[FitterBlockSize / LocalSize];
			#endif

			float tmp_sum_value = 0.f;
			for(int subVector = 0; subVector < FitterBlockSize / LocalSize; ++subVector)
			{
				const int index = subVector * LocalSize + threadId;
				if(index >= col_limited)
				{
					// Load feature

					#if USE_FEATURES_VGPR_CACHE
						const unsigned int featuresCacheOffset = baseFeaturesCacheOffset + subVector;
						#if USE_HALF_PRECISION_IN_FEATURES_DATA
						float tmp = HalfToFloat(featuresCache[featuresCacheOffset]);
						#else
						float tmp = featuresCache[featuresCacheOffset];
						#endif
					#else
						const unsigned int featureOffset = subVector * LocalSize + baseFeatureOffset;
						float tmp = load_feature(features_buffer, featureOffset);
					#endif

					// [Section 3.4] - Stochastic regularization
					// To handle rank-deficiency in the T matrix, Add zero-mean noise to the input buffers
					// (the first time values are loaded), which makes them linearly independent.
					// Note: does not Add noise to constant buffer (column 0) and noisy image data (last 3 columns).
					if(col == 0 && featureIndex < BUFFER_COUNT - 3)
					{
						const int seed = subVector * LocalSize + baseFeatureSeed;
						tmp += NOISE_AMOUNT * SignedZeroMeanNoise(seed);
					}

					#if CACHE_TMP_DATA
					tmp_data_private_cache[subVector] = tmp;
					#endif
					tmp_sum_value += tmp * u_vec[index];
				}
			}

			pSharedData[threadId] = tmp_sum_value;
			SyncThreads();
			parallel_reduction_sum<LocalSize>(&dotProd, pSharedData, threadId);

			const float dotFactor = 2.0f * dotProd / u_length_squared;

			// Manual unrolling for parallel reduction in case FitterBlockSize is greater than LocalSize
			// as the reduction operates on LocalSize elements.
			for(int subVector = 0; subVector < (FitterBlockSize / LocalSize); ++subVector)
			{
				const int index = subVector * LocalSize + threadId;
				if(index >= col_limited)
				{
					#if USE_FEATURES_VGPR_CACHE
					const unsigned int featuresCacheOffset = baseFeaturesCacheOffset + subVector;
					#else
					const unsigned int featureOffset = subVector * LocalSize + baseFeatureOffset;
					#endif

					#if CACHE_TMP_DATA
					float store_value = tmp_data_private_cache[subVector];
					#else
						#if USE_FEATURES_VGPR_CACHE
							#if USE_HALF_PRECISION_IN_FEATURES_DATA
							float store_value = HalfToFloat(featuresCache[featuresCacheOffset]);
							#else
							float store_value = featuresCache[featuresCacheOffset];
							#endif
						#else
							float store_value = load_feature(features_buffer, featureOffset);
						#endif
					const int seed = subVector * LocalSize + baseFeatureSeed;
					store_value += NOISE_AMOUNT * SignedZeroMeanNoise(seed);
					#endif 

					store_value -= dotFactor * u_vec[index];

					#if USE_FEATURES_VGPR_CACHE
						#if USE_HALF_PRECISION_IN_FEATURES_DATA
						featuresCache[featuresCacheOffset] = FloatToHalf(store_value);
						#else
						featuresCache[featuresCacheOffset] = store_value;
						#endif
					#else
						store_feature(features_buffer, featureOffset, store_value);
					#endif
				}
			}
			#if !USE_FEATURES_VGPR_CACHE
			GlobalMemFence();
			#endif
		}
	}

	// Back substitution
	__shared__ cvec3 divider; // Shared memory variable that holds the divider

	// R_EDGE = buffer_count - 2 (= number of features + 3 (noisy color spp buffer) - 2)
	// R is a (M + 1)x(M + 1) matrix, with M the number of features (here equal to buffer_count - 3)
	// which gives us R_EDGE = M + 1 = buffer_count - 3 + 1 = buffer_count - 2
	for(int i = R_EDGE - 2; i >= 0; i--)
	{
		if(threadId == 0)
			divider = load_r_mat(r_mat, i, i);
		
		SyncThreads();
		
		#if COMPRESSED_R
		if(threadId < R_EDGE && threadId >= i)
		#else
		// First values are always zero if R !COMPRESSED_R and
		// "&& threadId >= i" makes not compressed code run little bit slower
		if(threadId < R_EDGE)
		#endif
		{
			vec3 value = load_r_mat(r_mat, threadId, i);
			store_r_mat(r_mat, threadId, i, value / vec3(divider.x, divider.y, divider.z));
		}

		SyncThreads();

		#if 1 // ORIGINAL
		if(threadId == 0) // Optimization proposal: parallel reduction
		{
			for(int j = i + 1; j < R_EDGE - 1; j++)
			{
				vec3 value  = load_r_mat(r_mat, R_EDGE - 1, i);
				vec3 value2 = load_r_mat(r_mat, j, i);
				store_r_mat(r_mat, R_EDGE - 1, i, value - value2);
			}
		}
		#else
		const int startRIdx = (i + 1);
		const int endRIdx	= (R_EDGE - 1);
		const int numItems	= endRIdx - startRIdx;
		if(threadId < numItems)
		{
			// Parallel load
			const int j = startRIdx + threadId;
			vec3 value2 = load_r_mat(r_mat, j, i);

			// Then iterate over active threads and gather data from lane 0
			if(threadId == 0)
			{
				for(int k = startRIdx; k < endRIdx; k++)
				{
					vec3 value = load_r_mat(r_mat, R_EDGE - 1, i);
					vec3 currValue2 = ...;// load from lane k
					store_r_mat(r_mat, R_EDGE - 1, i, value - currValue2);
				}
			}
		}
		#endif

		SyncThreads();

		#if COMPRESSED_R
		if(threadId < R_EDGE && i >= threadId)
		#else
		if(threadId < R_EDGE)
		#endif
		{
			vec3 value  = load_r_mat(r_mat, i, threadId);
			vec3 value2 = load_r_mat(r_mat, R_EDGE - 1, i);
			store_r_mat(r_mat, i, threadId, value * value2);
		}
		SyncThreads();
	}

	// The features are stored in the first (buffers-3) values: the last 3 contain the noisy 1spp color channels
	if(threadId < BUFFER_COUNT - 3)
	{
		// Store weights
		const int index = groupId * (BUFFER_COUNT - 3) + threadId;
		const vec3 weight = load_r_mat(r_mat, R_EDGE - 1, threadId);
		store3(weights, index, weight);
	}
}

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
)
{
	switch(params.kernelLocalSize)
	{
		case 256:
		{
			switch(params.fitterBlockSize)
			{
				case 16: template_fitter<16 * 16, 256><<<grid_size, block_size>>>(params, weights, features_buffer); break;
				case 32: template_fitter<32 * 32, 256><<<grid_size, block_size>>>(params, weights, features_buffer); break;
				case 64: template_fitter<64 * 64, 256><<<grid_size, block_size>>>(params, weights, features_buffer); break;
				default: break;
			}
			break;
		}

		case 1024:
		{
			switch(params.fitterBlockSize)
			{
				case 32: template_fitter<32 * 32, 1024><<<grid_size, block_size>>>(params, weights, features_buffer); break;
				case 64: template_fitter<64 * 64, 1024><<<grid_size, block_size>>>(params, weights, features_buffer); break;
				default: break;
			}
			break;
		}

		default: break;
	}
}


inline __device__ void load_r_mat(const half * K_RESTRICT r_mat, unsigned int x, unsigned int y, half oValue[3])
{
	const unsigned int offset = 3 * R_ACCESS;
	oValue[0] = r_mat[offset + 0];
	oValue[1] = r_mat[offset + 1];
	oValue[2] = r_mat[offset + 2];
}

inline __device__ void store_r_mat(half * K_RESTRICT r_mat, unsigned int x, unsigned int y, half value[3])
{
	const unsigned int offset = 3 * R_ACCESS;
	r_mat[offset + 0] = value[0];
	r_mat[offset + 1] = value[1];
	r_mat[offset + 2] = value[2];
}

inline __device__ void store_r_mat_broadcast(half * r_mat, unsigned int x, unsigned int y, half value)
{
	const unsigned int offset = 3 * R_ACCESS;
	r_mat[offset + 0] = value;
	r_mat[offset + 1] = value;
	r_mat[offset + 2] = value;
}

inline __device__ void store_r_mat_channel(half * r_mat, unsigned int x, unsigned int y, unsigned int channel, half value)
{
	const unsigned int offset = 3 * R_ACCESS + channel;
	r_mat[offset] = value;
}

// Block size: (256, 1, 1)
__global__ void fitter16bits(
	FitterKernelParams params,
	//half * K_RESTRICT weights,			// [out] Features weights
	float * K_RESTRICT weights,			// [out] Features weights
	half * K_RESTRICT features_buffer	// [in]  Features buffer
)
{
	// Notes:
	//  LOCAL_SIZE = 256
	//	BLOCK_PIXELS = 32 * 32
	
	// TODO: send as define for cpp side
	#if COMPRESSED_R
    //const auto r_size = ((buffer_count - 2) * (buffer_count - 1) / 2) * sizeof(cl_float3);
	#define R_SHARED_DATA_SIZE ((BUFFER_COUNT - 2) * (BUFFER_COUNT - 1) / 2)
	#else
    //const auto r_size = (buffer_count - 2) * (buffer_count - 2) * sizeof(cl_float3);
	#define R_SHARED_DATA_SIZE ((BUFFER_COUNT - 2) * (BUFFER_COUNT - 2))
	#endif

	__shared__ half  pr_shared_data[LOCAL_SIZE];			// Shared memory used to perform parallel reduction
	__shared__ half  u_vec_sdata[BLOCK_PIXELS];				// Shared memory used to store the 'u' vectors
	__shared__ half  r_mat_sdata[3 * R_SHARED_DATA_SIZE];	// Shared memory used to store the R matrices of the QR factorization (x3 -> one per color channel)
	__shared__ half u_length_squared;						// Shared memory variable that holds the 'u' vector square length
	__shared__ half dotProd;								// Shared memory variable that holds the dot product of...
	__shared__ half vec_length;								// Shared memory variable that holds the vec length			

	half * pr_data_256 = &pr_shared_data[0];
	half * u_vec = &u_vec_sdata[0];
	half * r_mat = &r_mat_sdata[0];

	const int groupId = blockIdx.x;
	const int threadId = threadIdx.x; // in [0, 255]

	const unsigned int blockIndexX = groupId % params.worksetWithMarginBlockCountX;
	const unsigned int blockIndexY = groupId / params.worksetWithMarginBlockCountX;
	const unsigned int linearBlockIndex = blockIndexY * params.worksetWithMarginBlockCountX + blockIndexX;
	const unsigned int threadFeaturesBuffersOffset = linearBlockIndex * BUFFER_COUNT * BLOCK_PIXELS + threadId;

	const unsigned int baseSeed = params.frameNumber * BUFFER_COUNT * BLOCK_PIXELS + threadId;
	
	// Non square matrices require processing every column.
	// Otherwise result is OKish, but R is not upper triangular matrix
	const int limit = (BUFFER_COUNT == BLOCK_PIXELS) ? BUFFER_COUNT - 1 : BUFFER_COUNT;

	
	#if USE_FEATURES_VGPR_CACHE
	const unsigned int FeaturesCacheSize = BLOCK_PIXELS / LOCAL_SIZE * BUFFER_COUNT;
	half featuresCache[FeaturesCacheSize];

	for(unsigned int featureIndex = 0; featureIndex < BUFFER_COUNT; ++featureIndex)
	{
		const unsigned int baseFeatureOffset = featureIndex * BLOCK_PIXELS + threadFeaturesBuffersOffset;
		const unsigned int baseFeaturesCacheOffset = featureIndex * (BLOCK_PIXELS / LOCAL_SIZE);

		for(int subVector = 0; subVector < BLOCK_PIXELS / LOCAL_SIZE; ++subVector)
		{
			const unsigned int featureOffset = subVector * LOCAL_SIZE + baseFeatureOffset;
			const unsigned int featuresCacheOffset = baseFeaturesCacheOffset + subVector;
			featuresCache[featuresCacheOffset] = features_buffer[featureOffset];
		}
	}
	#endif

	// Compute R
	for(int col = 0; col < limit; col++)
	{
		// Note: the last 3 features values are the 3 channels of the color (not used for the regression)
		int col_limited = Min(col, BUFFER_COUNT - 3);

		// Load new column into memory
		const int featureIndex = col;
		
		#if USE_FEATURES_VGPR_CACHE
		const unsigned int baseFeaturesCacheOffset = featureIndex * (BLOCK_PIXELS / LOCAL_SIZE);
		#else
		const unsigned int baseFeatureOffset = featureIndex * BLOCK_PIXELS + threadFeaturesBuffersOffset;
		#endif

		half tmp_sum_value = FloatToHalf(0.0f);

		// Manual unrolling for parallel reduction as the block contains 1024 (32x32) work items and
		// the reduction operates on 256 elements (group size)
		// -> Compute the sum of N values (N = 1024/256 = 4)
		for(int subVector = 0; subVector < BLOCK_PIXELS / LOCAL_SIZE; ++subVector)
		{
			// Load feature
			#if USE_FEATURES_VGPR_CACHE
			const unsigned int featuresCacheOffset = baseFeaturesCacheOffset + subVector;
			half tmp = featuresCache[featuresCacheOffset];
			#else
			const unsigned int featureOffset = subVector * LOCAL_SIZE + baseFeatureOffset;
			half tmp = features_buffer[featureOffset];
			#endif

			// Store the feature in shared memory
			const int index = subVector * LOCAL_SIZE + threadId;
			u_vec[index] = tmp;

			if(index >= col_limited + 1)
			{
				tmp_sum_value = Add(tmp_sum_value, Mul(tmp, tmp));
			}
		}
		SyncThreads();

		// Find length of vector in A's column with reduction sum function
		pr_data_256[threadId] = tmp_sum_value;
		SyncThreads();
		parallel_reduction_sum_256(&vec_length, pr_data_256, threadId);

		// NOTE: GCN Opencl compiler can do some optimization with this because if
		// initially wanted col_limited is used to select wich work-item runs which branch
		// it is slower. However using col produces the same result.
		half r_value;
		if(threadId < col)
		{
			// Copy u_vec value
			r_value = u_vec[threadId];
		}
		else if(threadId == col)
		{
			u_length_squared = vec_length;
			vec_length = Sqrt(Add(vec_length, Mul(u_vec[col_limited], u_vec[col_limited])));
			u_vec[col_limited] = Sub(u_vec[col_limited], vec_length);
			u_length_squared = Add(u_length_squared, Mul(u_vec[col_limited], u_vec[col_limited]));

			// (u_length_squared is now updated length squared)
			r_value = vec_length;
		}
		else if(threadId > col) //Could have "&& threadId <  R_EDGE" but this is little bit faster
		{
			// Last values on every column are zeros
			r_value = FloatToHalf(0.0f);
		}

		int id_limited = Min(threadId, BUFFER_COUNT - 3);
		if(col < BUFFER_COUNT - 3)
			store_r_mat_broadcast(r_mat, col_limited, id_limited, r_value);
		else
			store_r_mat_channel(r_mat, col_limited, id_limited, col - BUFFER_COUNT + 3, r_value);
		SyncThreads();

		// Transform further columns of A
		// NOTE: three last columns are three color channels of noisy data. However,
		// they all need to be transfomed as they were column indexed (buffers - 3)
		for(int featureIndex = col_limited+1; featureIndex < BUFFER_COUNT; ++featureIndex)
		{
			#if USE_FEATURES_VGPR_CACHE
			const unsigned int baseFeaturesCacheOffset = featureIndex * (BLOCK_PIXELS / LOCAL_SIZE);
			#else
			const unsigned int baseFeatureOffset = featureIndex * BLOCK_PIXELS + threadFeaturesBuffersOffset;
			#endif

			const unsigned int baseFeatureSeed = featureIndex * BLOCK_PIXELS + baseSeed;

			// Starts by computing dot product with reduction sum function
			#if CACHE_TMP_DATA
			// No need to load features_buffer twice because each work-item first copies value for
			// dot product computation and then modifies the same value
			half tmp_data_private_cache[BLOCK_PIXELS / LOCAL_SIZE];
			#endif

			half tmp_sum_value = FloatToHalf(0.0f);
			for(int subVector = 0; subVector < BLOCK_PIXELS / LOCAL_SIZE; ++subVector)
			{
				const int index = subVector * LOCAL_SIZE + threadId;
				if(index >= col_limited)
				{
					// Load feature
					#if USE_FEATURES_VGPR_CACHE
					const unsigned int featuresCacheOffset = baseFeaturesCacheOffset + subVector;
					half tmp = featuresCache[featuresCacheOffset];
					#else
					const unsigned int featureOffset = subVector * LOCAL_SIZE + baseFeatureOffset;
					half tmp = features_buffer[featureOffset];
					#endif

					// [Section 3.4] - Stochastic regularization
					// To handle rank-deficiency in the T matrix, Add zero-mean noise to the input buffers
					// (the first time values are loaded), which makes them linearly independent.
					// Note: does not Add noise to constant buffer (column 0) and noisy image data (last 3 columns).
					if(col == 0 && featureIndex < BUFFER_COUNT - 3)
					{
						const int seed = subVector * LOCAL_SIZE + baseFeatureSeed;
						tmp = Add(tmp, FloatToHalf(NOISE_AMOUNT * SignedZeroMeanNoise(seed)));
					}

					#if CACHE_TMP_DATA
					tmp_data_private_cache[subVector] = tmp;
					#endif
					tmp_sum_value = Add(tmp_sum_value, Mul(tmp, u_vec[index]));
				}
			}

			pr_data_256[threadId] = tmp_sum_value;
			SyncThreads();
			parallel_reduction_sum_256(&dotProd, pr_data_256, threadId);

			const half dotFactor = Mul(FloatToHalf(2.0f), Div(dotProd, u_length_squared));

			// Manual unrolling as the block contains 1024 (32x32) work items and we operate on 256 elements (group size)
			// -> Compute the sum of N values (N = 1024/256 = 4)
			for(int subVector = 0; subVector < BLOCK_PIXELS / LOCAL_SIZE; ++subVector)
			{
				const int index = subVector * LOCAL_SIZE + threadId;
				if(index >= col_limited)
				{
					#if USE_FEATURES_VGPR_CACHE
					const unsigned int featuresCacheOffset = baseFeaturesCacheOffset + subVector;
					#else
					const unsigned int featureOffset = subVector * LOCAL_SIZE + baseFeatureOffset;
					#endif

					#if CACHE_TMP_DATA
						half store_value = tmp_data_private_cache[subVector];
					#else
						#if USE_FEATURES_VGPR_CACHE
						half store_value = featuresCache[featuresCacheOffset];
						#else
						half store_value = features_buffer[featureOffset];
						#endif
						const int seed = subVector * LOCAL_SIZE + baseFeatureSeed;
						store_value = Add(store_value, FloatToHalf(NOISE_AMOUNT * SignedZeroMeanNoise(seed)));
					#endif 

					store_value = Sub(store_value, Mul(dotFactor, u_vec[index]));

					#if USE_FEATURES_VGPR_CACHE
					featuresCache[featuresCacheOffset] = store_value;
					#else
					features_buffer[featureOffset] = store_value;
					#endif
				}
			}
			#if !USE_FEATURES_VGPR_CACHE
			GlobalMemFence();
			#endif
		}
	}

	// Back substitution
	__shared__ half divider[3]; // Shared memory variable that holds the divider

	// R_EDGE = buffer_count - 2 (= number of features + 3 (noisy color spp buffer) - 2)
	// R is a (M + 1)x(M + 1) matrix, with M the number of features (here equal to buffer_count - 3)
	// which gives us R_EDGE = M + 1 = buffer_count - 3 + 1 = buffer_count - 2
	for(int i = R_EDGE - 2; i >= 0; i--)
	{
		if(threadId == 0)
			load_r_mat(r_mat, i, i, divider);
		
		SyncThreads();
		
		#if COMPRESSED_R
		if(threadId < R_EDGE && threadId >= i)
		#else
		// First values are always zero if R !COMPRESSED_R and
		// "&& threadId >= i" makes not compressed code run little bit slower
		if(threadId < R_EDGE)
		#endif
		{
			half value[3];
			load_r_mat(r_mat, threadId, i, value);
			Div(value, divider, value);
			store_r_mat(r_mat, threadId, i, value);
		}

		SyncThreads();

		if(threadId == 0) // Optimization proposal: parallel reduction
		{
			for(int j = i + 1; j < R_EDGE - 1; j++)
			{
				half value[3];
				load_r_mat(r_mat, R_EDGE - 1, i, value);
				half value2[3];
				load_r_mat(r_mat, j, i, value2);
				Sub(value, value2, value);
				store_r_mat(r_mat, R_EDGE - 1, i, value);
			}
		}

		SyncThreads();

		#if COMPRESSED_R
		if(threadId < R_EDGE && i >= threadId)
		#else
		if(threadId < R_EDGE)
		#endif
		{
			half value[3];
			load_r_mat(r_mat, i, threadId, value);
			half value2[3];
			load_r_mat(r_mat, R_EDGE - 1, i, value2);
			Mul(value, value2, value);
			store_r_mat(r_mat, i, threadId, value);
		}
		SyncThreads();
	}

	// The features are stored in the first (buffers-3) values: the last 3 contain the noisy 1spp color channels
	if(threadId < BUFFER_COUNT - 3)
	{
		// Store weights
		const int index = groupId * (BUFFER_COUNT - 3) + threadId;
		half weight[3];
		load_r_mat(r_mat, R_EDGE - 1, threadId, weight);
		vec3 fweight;
		fweight.x = HalfToFloat(weight[0]);
		fweight.y = HalfToFloat(weight[1]);
		fweight.z = HalfToFloat(weight[2]);
		store3(weights, index, fweight);
	}
}


extern "C" void run_fitter16bits(
	dim3 const & grid_size,
	dim3 const & block_size,
	FitterKernelParams params,
	//half * K_RESTRICT weights,			// [out] Features weights
	float * K_RESTRICT weights,			// [out] Features weights
	half * K_RESTRICT features_buffer	// [in]  Features buffer
)
{
	fitter16bits<<<grid_size, block_size>>>(params, weights, features_buffer);
}

// Weighted sum kernel /////////////////////////////////////////////////////////
// -> outputs the noise-free 1spp color estimate

template <int FitterBlockSize>
__global__ void template_weighted_sum(
	WeightedSumKernelParams params,
	const float * K_RESTRICT weights,			// [in]	 Features weights computed by the fitter kernel
		  float * K_RESTRICT output,			// [out] Noise-free color estimate
	const float * K_RESTRICT current_normals,	// [in]  Current (world) normals
	const float * K_RESTRICT normalized_world_positions	// [in]  Current world positions
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
	const int group_index = (offset_pixel.x / FitterBlockSize) + (offset_pixel.y / FitterBlockSize) * params.worksetWithMarginBlockCountX;

	// Reload features from buffer here to have values without stochastic regularization noise
	// TODO: bind the normalized world_position buffer to avoid renormalizing again (no need for mins_maxs buffer)
	vec3 normalized_world_position = load3<float>(normalized_world_positions, linear_pixel); 
	vec3 normal = load3<float>(current_normals, linear_pixel);

	float features[BUFFER_COUNT-3];
	compute_features_without_color(normalized_world_position, normal, features);

	const unsigned baseWeightOffset = group_index * (BUFFER_COUNT - 3);

	// Weighted sum of the feature buffers
	vec3 color = vec3(0.f, 0.f, 0.f);
	for(int feature_buffer = 0; feature_buffer < BUFFER_COUNT - 3; feature_buffer++)
	{
		float feature = features[feature_buffer];
		vec3 weight = load3<float>(weights, baseWeightOffset + feature_buffer);
		color += weight * feature;
	}

	// Remove negative values from every component of the fitting results
	color = Max(vec3(0.f), color); // TODO -Min(-color, vec3(0.f));

	// Store results
	store3(output, linear_pixel, color);
}

extern "C" void run_new_weighted_sum(
	dim3 const & grid_size,
	dim3 const & block_size,
	WeightedSumKernelParams const & params,
	const float * K_RESTRICT weights,			// [in]	 Features weights computed by the fitter kernel
		  float * K_RESTRICT output,			// [out] Noise-free color estimate
	const float * K_RESTRICT current_normals,	// [in]  Current (world) normals
	const float * K_RESTRICT current_positions	// [in]  Current world positions
)
{
	switch(params.fitterBlockSize)
	{
		case 16:
		{
			template_weighted_sum<16><<<grid_size, block_size>>>(
				params,
				weights,
				output,
				current_normals,
				current_positions
			);
			break;
		}

		case 32:
		{
			template_weighted_sum<32><<<grid_size, block_size>>>(
				params,
				weights,
				output,
				current_normals,
				current_positions
			);
			break;
		}

		case 64:
		{
			template_weighted_sum<64><<<grid_size, block_size>>>(
				params,
				weights,
				output,
				current_normals,
				current_positions
			);
			break;
		}

		default: break;
	}
}

// Accumulate filtered data kernel /////////////////////////////////////////////
// -> outputs the noise-free accumulated color estimate + a tonemapped version w/ albedo

__global__ void accumulate_filtered_data_frame0(
	AccumulateFilteredDataKernelParams2 params,
	const float * K_RESTRICT filtered_frame,			// [in]  Noise free color estimate (computed as the weighted sum of the features)
	const float * K_RESTRICT albedo_buffer,				// [in]  Albedo buffer of the current frame (non-noisy)
		  float * K_RESTRICT tone_mapped_frame,			// [out] Accumulated and tonemapped noise-free color estimate
		  float * K_RESTRICT accumulated_frame			// [out] Current frame noise-free accumulated color estimate
)
{
	const ivec2 pixel = ivec2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	
	const int w = params.sizeX;
	const int h = params.sizeY;
	
	if(pixel.x >= w || pixel.y >= h)
		return;

	// Linear pixel index
	const unsigned int linear_pixel = pixel.y * w + pixel.x;

	// Noise-free estimate of the color (computed via a weighted sum of features)
	vec3 filtered_color = load3<float>(filtered_frame, linear_pixel);
	store3(accumulated_frame, linear_pixel, filtered_color);

	// Remodulate albedo and tone map
	vec3 albedo = load3<float>(albedo_buffer, linear_pixel);
	const vec3 tone_mapped_color = Clamp(Pow(Max(vec3(0.f), albedo * filtered_color), 0.454545f), vec3(0.f), vec3(1.f));
	store3(tone_mapped_frame, linear_pixel, tone_mapped_color);
}

extern "C" void run_accumulate_filtered_data_frame0(
	dim3 const & grid_size,
	dim3 const & block_size,
	AccumulateFilteredDataKernelParams2 const & params,
	const float * K_RESTRICT filtered_frame,			// [in]  Noise free color estimate (computed as the weighted sum of the features)
	const float * K_RESTRICT albedo_buffer,				// [in]  Albedo buffer of the current frame (non-noisy)
		  float * K_RESTRICT tone_mapped_frame,			// [out] Accumulated and tonemapped noise-free color estimate
		  float * K_RESTRICT accumulated_frame			// [out] Current frame noise-free accumulated color estimate
)
{
	accumulate_filtered_data_frame0<<<grid_size, block_size>>>(
		params,
		filtered_frame,
		albedo_buffer,
		tone_mapped_frame,
		accumulated_frame
	);
}


__global__ void new_accumulate_filtered_data(
	AccumulateFilteredDataKernelParams2 params,
	const float * K_RESTRICT filtered_frame,			// [in]  Noise free color estimate (computed as the weighted sum of the features)
	const vec2 * K_RESTRICT in_prev_frame_pixel,		// [in]  Previous frame pixel coordinates (after reprojection)
	const unsigned char * K_RESTRICT accept_bools,		// [in]  Validity mask of bilinear samples in previous frame (after reprojection)
	const float * K_RESTRICT albedo_buffer,				// [in]  Albedo buffer of the current frame (non-noisy)
		  float * K_RESTRICT tone_mapped_frame,			// [out] Accumulated and tonemapped noise-free color estimate
	const unsigned char* K_RESTRICT current_spp,		// [in]	 Current number of samples accumulated (for CMA)
	const float * K_RESTRICT accumulated_prev_frame,	// [in]  Previous frame noise-free accumulated color estimate 
		  float * K_RESTRICT accumulated_frame			// [out] Current frame noise-free accumulated color estimate
)
{
	const ivec2 pixel = ivec2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	
	const int w = params.sizeX;
	const int h = params.sizeY;
	
	if(pixel.x >= w || pixel.y >= h)
		return;

	// Linear pixel index
	const unsigned int linear_pixel = pixel.y * w + pixel.x;

	// Noise-free estimate of the color (computed via a weighted sum of features)
	vec3 filtered_color = load3<float>(filtered_frame, linear_pixel);
	vec3 prev_color = vec3(0.f, 0.f, 0.f);
	float blend_alpha = 1.f;

	// Reproject and accumulate previous frame noise-free estimate

	// Bitmask telling which bilinear samples were accepted in the first accumulation kernel
	const unsigned char accept = accept_bools[linear_pixel];

	if(accept > 0) // If any prev frame sample is accepted
	{
		// Pixel coordinates in the previous frame (in [0, w-1]x[0, h-1])
		const vec2 prev_frame_pixel_f = in_prev_frame_pixel[linear_pixel];
			
		// Integer pixel coordinates in the previous frame
		const ivec2 prev_frame_pixel_i = FloatToIntRd(prev_frame_pixel_f);

		// Compute bilinear weights for bilinear sampling
		const vec2 prev_pixel_fract = prev_frame_pixel_f - vec2(prev_frame_pixel_i);
		const vec2 one_minus_prev_pixel_fract = 1.f - prev_pixel_fract;

		float total_weight = 0.f;

		// Add valid bilinear samples

		if(accept & 0x01)
		{
			float weight = one_minus_prev_pixel_fract.x * one_minus_prev_pixel_fract.y;
			int linear_sample_location = prev_frame_pixel_i.y * w + prev_frame_pixel_i.x;
			prev_color += weight * load3<float>(accumulated_prev_frame, linear_sample_location);
			total_weight += weight;
		}

		if(accept & 0x02)
		{
			float weight = prev_pixel_fract.x * one_minus_prev_pixel_fract.y;
			int linear_sample_location = prev_frame_pixel_i.y * w + prev_frame_pixel_i.x + 1;
			prev_color += weight * load3<float>(accumulated_prev_frame, linear_sample_location);
			total_weight += weight;
		}

		if(accept & 0x04)
		{
			float weight = one_minus_prev_pixel_fract.x * prev_pixel_fract.y;
			int linear_sample_location = (prev_frame_pixel_i.y + 1) * w + prev_frame_pixel_i.x;
			prev_color += weight * load3<float>(accumulated_prev_frame, linear_sample_location);
			total_weight += weight;
		}

		if(accept & 0x08)
		{
			float weight = prev_pixel_fract.x * prev_pixel_fract.y;
			int linear_sample_location = (prev_frame_pixel_i.y + 1) * w + prev_frame_pixel_i.x + 1;
			prev_color += weight * load3<float>(accumulated_prev_frame, linear_sample_location);
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
			blend_alpha = 1.f / float(current_spp[linear_pixel]);
			blend_alpha = Max(blend_alpha, SECOND_BLEND_ALPHA);
			prev_color /= total_weight;

			// Note: we accumulate at most 255 samples for the cumulative moving average (which is more than enough because of
			// the threshold SECOND_BLEND_ALPHA that switch to exponential moving average).
			// E.g: SECOND_BLEND_ALPHA = 0.1 = 1 / (n + 1) <=> n = (1 - 0.1) / 0.1 = 9 => above 9 samples for a pixel,
			// we switch to exponential moving average with alpha = 10%
		}
	}

	// Mix with colors and store results
	vec3 accumulated_color = blend_alpha * filtered_color + (1.f - blend_alpha) * prev_color; // Lerp(prev_color, filtered_color, blend_alpha);
	store3(accumulated_frame, linear_pixel, accumulated_color);

	// Remodulate albedo and tone map
	vec3 albedo = load3<float>(albedo_buffer, linear_pixel);
	const vec3 tone_mapped_color = Clamp(Pow(Max(vec3(0.f), albedo * accumulated_color), 0.454545f), vec3(0.f), vec3(1.f));
	store3(tone_mapped_frame, linear_pixel, tone_mapped_color);
}

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
)
{
	new_accumulate_filtered_data<<<grid_size, block_size>>>(
		params,
		filtered_frame,
		in_prev_frame_pixel,
		accept_bools,
		albedo_buffer,
		tone_mapped_frame,
		current_spp,
		accumulated_prev_frame,
		accumulated_frame
	);
}


// TAA kernel //////////////////////////////////////////////////////////////////

// TODO:
// - optimize with local/shared memory
__global__ void taa_frame0(
	TAAKernelParams params,
	const float * K_RESTRICT new_frame,				// [in]	 Current frame color buffer
		  float * K_RESTRICT result_frame			// [out] Antialiased frame color buffer
)
{
	const ivec2 pixel = ivec2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
   
	const int w = params.sizeX;
	const int h = params.sizeY;

	if(pixel.x >= w || pixel.y >= h)
		return;

	// Linear pixel index
	const unsigned int linear_pixel = pixel.y * w + pixel.x;

	store3(result_frame, linear_pixel, load3<float>(new_frame, linear_pixel));
}

extern "C" void run_taa_frame0(
	dim3 const & grid_size,
	dim3 const & block_size,
	TAAKernelParams const & params,
	const float * K_RESTRICT new_frame,				// [in]	 Current frame color buffer
		  float * K_RESTRICT result_frame			// [out] Antialiased frame color buffer
)
{
	taa_frame0<<<grid_size, block_size>>>(params, new_frame, result_frame);
}


__global__ void new_taa(
	TAAKernelParams params,
	const vec2 * K_RESTRICT in_prev_frame_pixel,	// [in]  Previous frame pixel coordinates (after reprojection)
	const float * K_RESTRICT new_frame,				// [in]	 Current frame color buffer
		  float * K_RESTRICT result_frame,			// [out] Antialiased frame color buffer
	const float * K_RESTRICT prev_frame				// [in]  Previous frame color buffer
)
{
	const ivec2 pixel = ivec2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
   
	const int w = params.sizeX;
	const int h = params.sizeY;

	if(pixel.x >= w || pixel.y >= h)
		return;

	// Linear pixel index
	const unsigned int linear_pixel = pixel.y * w + pixel.x;

	// Current frame color
	vec3 my_new_color =	load3<float>(new_frame, linear_pixel);

	// Previous frame pixel coordinates
	const vec2 prev_frame_pixel_f = in_prev_frame_pixel[linear_pixel];
	const ivec2 prev_frame_pixel_i = FloatToIntRd(prev_frame_pixel_f);

	// Return if all sampled pixels are going to be out of image area
	if(prev_frame_pixel_i.x < -1 || prev_frame_pixel_i.y < -1 ||
	   prev_frame_pixel_i.x >= w || prev_frame_pixel_i.y >= h
	)
	{
		store3(result_frame, linear_pixel, my_new_color);
		return;
	}

	// Compute the color AABB in the 3x3 neighbourhood and the min/max in a cross pattern around the current pixel
	vec3 minimum_box	= vec3(+C_FLT_MAX);
	vec3 minimum_cross	= vec3(+C_FLT_MAX);
	vec3 maximum_box	= vec3(-C_FLT_MAX);
	vec3 maximum_cross	= vec3(-C_FLT_MAX);
	for(int y = -1; y <= 1; ++y)
	{
		for(int x = -1; x <= 1; ++x)
		{
			ivec2 sample_location = pixel + ivec2(x, y);
			if(sample_location.x >= 0 && sample_location.y >= 0 &&
			   sample_location.x < w && sample_location.y < h
			)
			{
				vec3 sample_color;
				if(x == 0 && y == 0)
					sample_color = my_new_color;
				else
					sample_color = load3<float>(new_frame, sample_location.x + sample_location.y * w);

				sample_color = RGB_to_YCoCg(sample_color);

				if(x == 0 || y == 0)
				{
					minimum_cross = Min(minimum_cross, sample_color);
					maximum_cross = Max(maximum_cross, sample_color);
				}

				minimum_box = Min(minimum_box, sample_color);
				maximum_box = Max(maximum_box, sample_color);
			}
		}
	}

	// Bilinear sampling of previous frame.
	// Note: work-item has already returned if the sampling location is completly out of image
	vec3 prev_color = vec3(0.f, 0.f, 0.f);
	float total_weight = 0;
	const vec2 pixel_fract = prev_frame_pixel_f - vec2(prev_frame_pixel_i);
	const vec2 one_minus_pixel_fract = 1.f - pixel_fract;

	if(prev_frame_pixel_i.y >= 0)
	{
		if(prev_frame_pixel_i.x >= 0)
		{
			float weight = one_minus_pixel_fract.x * one_minus_pixel_fract.y;
			prev_color += weight * load3<float>(prev_frame, prev_frame_pixel_i.y * w + prev_frame_pixel_i.x);
			total_weight += weight;
		}

		if(prev_frame_pixel_i.x < w - 1)
		{
			float weight = pixel_fract.x * one_minus_pixel_fract.y;
			prev_color += weight * load3<float>(prev_frame, prev_frame_pixel_i.y * w + prev_frame_pixel_i.x + 1);
			total_weight += weight;
		}
	}

	if(prev_frame_pixel_i.y < h - 1)
	{
		if(prev_frame_pixel_i.x >= 0)
		{
			float weight = one_minus_pixel_fract.x * pixel_fract.y;
			prev_color += weight * load3<float>(prev_frame, (prev_frame_pixel_i.y + 1) * w + prev_frame_pixel_i.x);
			total_weight += weight;
		}

		if(prev_frame_pixel_i.x < w - 1)
		{
			float weight = pixel_fract.x * pixel_fract.y;
			prev_color += weight * load3<float>(prev_frame, (prev_frame_pixel_i.y + 1) * w + prev_frame_pixel_i.x + 1);
			total_weight += weight;
		}
	}

	if(total_weight > 0)
		prev_color /= total_weight; // Total weight can be less than one on the edges

	vec3 prev_color_ycocg = RGB_to_YCoCg(prev_color);

	// Note: Some references use more complicated methods to move the previous frame color to the YCoCg space AABB
	vec3 minimum = (minimum_box + minimum_cross) / 2.f;
	vec3 maximum = (maximum_box + maximum_cross) / 2.f;
	vec3 prev_color_rgb = YCoCg_to_RGB(Clamp(prev_color_ycocg, minimum, maximum));

	vec3 result_color = TAA_BLEND_ALPHA * my_new_color + (1.f - TAA_BLEND_ALPHA) * prev_color_rgb; // Lerp(prev_color_rgb, my_new_color, TAA_BLEND_ALPHA);
	store3(result_frame, linear_pixel, result_color);
}

extern "C" void run_new_taa(
	dim3 const & grid_size,
	dim3 const & block_size,
	TAAKernelParams const & params,
	const vec2 * K_RESTRICT in_prev_frame_pixel,	// [in]  Previous frame pixel coordinates (after reprojection)
	const float * K_RESTRICT new_frame,				// [in]	 Current frame color buffer
		  float * K_RESTRICT result_frame,			// [out] Antialiased frame color buffer
	const float * K_RESTRICT prev_frame				// [in]  Previous frame color buffer
)
{
	new_taa<<<grid_size, block_size>>>(
		params,
		in_prev_frame_pixel,
		new_frame,
		result_frame,
		prev_frame
	);
}
