#pragma once

#include <CL/cl.h>
#include <memory>

#include "utils.hpp"


#define K_OPENCL_CHECK(openclFunc) \
	do { \
		/*printf(#cudaFunc "\n");\*/ \
		cl_int ret = openclFunc; \
		if(ret != CL_SUCCESS) \
		{ \
			printf("OpenCL error: %d\n", ret); \
			__debugbreak(); \
		} \
	} while(0)


class OpenCLDeviceBuffer
{
	public:
		OpenCLDeviceBuffer(): m_data(nullptr), m_size(0)
		{
	
		}

		OpenCLDeviceBuffer(cl_context c, cl_mem_flags f, size_t s)
		{
			init(c, f, s);
		}

		~OpenCLDeviceBuffer()
		{
			destroy();
		}

		void init(cl_context c, cl_mem_flags f, size_t s)
		{
			if(m_data)
			{
				destroy();
			}

			cl_int ret;
			cl_mem data = clCreateBuffer(c, f, s, nullptr, &ret);
			if(ret == CL_SUCCESS)
			{
				m_data = data;
				m_size = s;
			}
		}

		void destroy()
		{
			if(m_data)
			{
				K_OPENCL_CHECK(clReleaseMemObject(m_data));
				m_data = nullptr;
				m_size = 0;
			}
		}

		cl_mem * data() { return &m_data; }
		cl_mem const * data() const { return &m_data; }
		size_t size() const { return m_size; }

	private:
		cl_mem m_data;
		size_t m_size;
};


// Only work with 3 channels float32 image
static void SaveDevice3Float32ImageToDisk(
	std::string const & filename,
	int frame,
	cl_command_queue const & command_queue,
	cl_mem const & buffer,
	BufferDesc const & desc
)
{
	const size_t datasize = desc.byte_size;
	const size_t numelem  = datasize / sizeof(float);
	std::vector<float> outdata;
	outdata.resize(numelem);

	K_OPENCL_CHECK(clEnqueueReadBuffer(command_queue, buffer, false, 0, datasize, outdata.data(), 0, NULL, NULL));
	K_OPENCL_CHECK(clFlush(command_queue));
	K_OPENCL_CHECK(clFinish(command_queue));

	std::string output_filename = OUTPUT_FOLDER + filename + "_" + std::to_string(frame) + "_opencl.png";

	// Output image
	printf("  Save image %s\n", output_filename.c_str());

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
        printf("  Can't create image file on disk to location %s\n", output_filename.c_str());
    }
}