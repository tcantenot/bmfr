#pragma once

#include <OpenImageDenoise/oidn.hpp>

#include "utils.hpp"


// Only work with 3-channel float32 image
inline void SaveDevice3Float32ImageToDisk(
	std::string const & filename,
	unsigned int frame,
	float * buffer,
	size_t w,
	size_t h,
	char const * suffix = "_cuda.png"
)
{
	const size_t xstride  = 3 * sizeof(float);
	const size_t ystride  = w * xstride;
	const size_t datasize = h * ystride;

	std::string output_filename = OUTPUT_FOLDER + filename + "_" + std::to_string(frame) + suffix;

	// Output image
	LOG("  Save image %s\n", output_filename.c_str());

    OpenImageIO::ImageSpec spec(w, h, 3, OpenImageIO::TypeDesc::FLOAT);
    std::unique_ptr<OpenImageIO::ImageOutput> out(OpenImageIO::ImageOutput::create(output_filename));
    if(out && out->open(output_filename, spec))
    {
        out->write_image(OpenImageIO::TypeDesc::FLOAT, buffer, xstride, ystride, 0);
        out->close();
    }
    else
    {
        LOG("  Can't create image file on disk to location %s\n", output_filename.c_str());
    }
}

inline void IntelOpenImageDenoiser()
{
	// Create an Open Image Denoise device
	OIDNDevice device = oidnNewDevice(OIDN_DEVICE_TYPE_DEFAULT);
	oidnCommitDevice(device);

	size_t width  = IMAGE_WIDTH;
	size_t height = IMAGE_HEIGHT;
	FrameInputData frameData;

	OIDNFilter filter = oidnNewFilter(device, "RT"); // generic ray tracing filter
	
	float * outputPtr = new float[width * height * 3];

	for(unsigned int frame = 59; frame < 60; ++frame)
	{
		LoadFrameInputData(frameData, width, height, frame);

		float * colorPtr  = frameData.noisy1spps.data();
		float * albedoPtr = frameData.albedos.data();
		float * normalPtr = frameData.normals.data();

		// Add albedo back to color input that is 1spp w/o albedo
		for(size_t i = 0, n = width * height * 3; i < n; ++i)
		{
			colorPtr[i] *= albedoPtr[i];
		}

		// Create a denoising filter
		oidnSetSharedFilterImage(filter, "color",  colorPtr,  OIDN_FORMAT_FLOAT3, width, height, 0, 0, 0);
		oidnSetSharedFilterImage(filter, "albedo", albedoPtr, OIDN_FORMAT_FLOAT3, width, height, 0, 0, 0); // optional
		oidnSetSharedFilterImage(filter, "normal", normalPtr, OIDN_FORMAT_FLOAT3, width, height, 0, 0, 0); // optional
		oidnSetSharedFilterImage(filter, "output", outputPtr, OIDN_FORMAT_FLOAT3, width, height, 0, 0, 0);
		oidnSetFilter1b(filter, "hdr", true); // image is HDR
		oidnCommitFilter(filter);

		// Filter the image
		oidnExecuteFilter(filter);

		// Check for errors
		const char* errorMessage;
		if (oidnGetDeviceError(device, &errorMessage) != OIDN_ERROR_NONE)
		  printf("Error: %s\n", errorMessage);

		// "Tonemap" image for comparison
		for(size_t i = 0, n = width * height * 3; i < n; ++i)
		{
			outputPtr[i] = std::min(pow(std::max(0.0f, outputPtr[i]), 0.454545f), 1.0f);
		}

		// Save image to disk
		SaveDevice3Float32ImageToDisk("result", frame, outputPtr, width, height, "_oidn.png");
	}

	// Cleanup
	delete[] outputPtr;
	oidnReleaseFilter(filter);
	oidnReleaseDevice(device);
}