#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cassert>
#include <memory>

#include "config.hpp"
#include "utils.hpp"


#define K_CUDA_CHECK(cudaFunc) \
	do { \
		/*LOG(#cudaFunc "\n");\*/ \
		cudaError_t ret = cudaFunc; \
		if(ret != cudaSuccess) \
		{ \
			LOG("Cuda error: %d\n", ret); \
			__debugbreak(); \
		} \
	} while(0)


class CudaDeviceBuffer
{
	public:
		CudaDeviceBuffer():	m_data(nullptr), m_size(0)
		{
	
		}

		CudaDeviceBuffer(size_t s)
		{
			init(s);
		}

		~CudaDeviceBuffer()
		{
			destroy();
		}

		void init(size_t s)
		{
			if(m_data)
			{
				destroy();
			}

			K_CUDA_CHECK(cudaMalloc(&m_data, s));
			m_size = s;
		}

		void destroy()
		{
			if(m_data)
			{
				K_CUDA_CHECK(cudaFree(m_data));
				m_data = nullptr;
				m_size = 0;
			}
		}

		void * data() { return m_data; }
		void const * data() const { return m_data; }
		size_t size() const { return m_size; }

		template <typename T>
		T * getTypedData() { return static_cast<T*>(m_data); }

		template <typename T>
		T const * getTypedData() const { return static_cast<T const*>(m_data); }

	private:
		void * m_data;
		size_t m_size;
};

class CudaTimer
{
	public:
		CudaTimer()
		{
			K_CUDA_CHECK(cudaEventCreate(&m_start));
			K_CUDA_CHECK(cudaEventCreate(&m_stop));
		}

		~CudaTimer()
		{
			K_CUDA_CHECK(cudaEventDestroy(m_start));
			K_CUDA_CHECK(cudaEventDestroy(m_stop));
		}

		void start()
		{
			K_CUDA_CHECK(cudaEventRecord(m_start));
		}

		void stop()
		{
			K_CUDA_CHECK(cudaEventRecord(m_stop));
		}

		float elaspedTime()
		{
			K_CUDA_CHECK(cudaEventSynchronize(m_stop));
			float elaspedTime_ms = 0.f;
			K_CUDA_CHECK(cudaEventElapsedTime(&elaspedTime_ms, m_start, m_stop));
			return elaspedTime_ms;
		}

	private:
		cudaEvent_t m_start;
		cudaEvent_t m_stop;
};


// Only work with 3-channel float32 image
inline void SaveDevice3Float32ImageToDisk(
	std::string const & filename,
	int frame,
	CudaDeviceBuffer const & buffer,
	BufferDesc const & desc,
	char const * suffix = "_cuda.png"
)
{
	assert(buffer.size() == desc.w * desc.h * 3 * sizeof(float));
	const size_t datasize = desc.byte_size;
	const size_t numelem  = datasize / sizeof(float);
	std::vector<float> outdata;
	outdata.resize(numelem);

	K_CUDA_CHECK(cudaMemcpy(outdata.data(), buffer.data(), datasize, cudaMemcpyDeviceToHost));
	K_CUDA_CHECK(cudaDeviceSynchronize());

	std::string output_filename = OUTPUT_FOLDER + filename + "_" + std::to_string(frame) + suffix;

	// Output image
	LOG("  Save image %s\n", output_filename.c_str());

	const int w = static_cast<int>(desc.w);
	const int h = static_cast<int>(desc.h);
    OpenImageIO::ImageSpec spec(w, h, 3, OpenImageIO::TypeDesc::FLOAT);
    std::unique_ptr<OpenImageIO::ImageOutput> out(OpenImageIO::ImageOutput::create(output_filename));
    if(out && out->open(output_filename, spec))
    {
        out->write_image(OpenImageIO::TypeDesc::FLOAT, outdata.data(), desc.x_stride, desc.y_stride, 0);
        out->close();
    }
    else
    {
        LOG("  Can't create image file on disk to location %s\n", output_filename.c_str());
    }
}
