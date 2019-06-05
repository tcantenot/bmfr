#pragma once

#include <string>

#include "OpenImageIO/imageio.h"

#include "config.hpp"


// Creates two same buffers and swap() call can be used to change which one is considered
// current and which one previous
template <class T>
class Double_buffer
{
    private:
        T a, b;
        bool swapped;

    public:
		Double_buffer(T aa, T bb): a(aa), b(bb) { }
        template <typename... Args>
        Double_buffer(Args && ...args) : a(std::forward<Args>(args)...), b(std::forward<Args>(args)...), swapped(false){ }
        T & current()  { return swapped ? a : b; }
        T const & current() const { return swapped ? a : b; }
        T & previous() { return swapped ? b : a; }
		T & operator[](unsigned int i) { return (i == 0) ? a : b; }
		T const & operator[](unsigned int i) const { return (i == 0) ? a : b; }
        void swap() { swapped = !swapped; }
};

////////////////////////////////////////////////////////////////////////////////

struct Operation_result
{
    bool success;
    std::string error_message;
    Operation_result(bool success, std::string const & error_message = "") :
        success(success), error_message(error_message) {}
};

static Operation_result read_image_file(std::string const & file_name, const int frame, float * buffer)
{
    OpenImageIO::ImageInput * in = OpenImageIO::ImageInput::open(file_name + std::to_string(frame) + ".exr");
    if(!in || in->spec().nchannels != 3)
    {
        return { false, "Can't open image file or it has wrong type: " + file_name };
    }

    // NOTE: this converts .exr files that might be in halfs to single precision floats
    // In the dataset distributed with the BMFR paper all exr files are in single precision
    in->read_image(OpenImageIO::TypeDesc::FLOAT, buffer);
    in->close();

    return { true };
}

static Operation_result load_image(float *image, const std::string file_name, const int frame)
{
    Operation_result result = read_image_file(file_name, frame, image);
    if(!result.success)
        return result;

    return {true};
}

////////////////////////////////////////////////////////////////////////////////

struct FrameInputData
{
	std::vector<float> albedos;
	std::vector<float> normals;
	std::vector<float> positions;
	std::vector<float> noisy1spps;
};

inline bool LoadFrameInputData(FrameInputData & frameInputData, size_t w, size_t h, int frame)
{
	// Allocate frame input data buffers
	frameInputData.albedos.resize(3 * w * h);
	frameInputData.normals.resize(3 * w * h);
	frameInputData.positions.resize(3 * w * h);
	frameInputData.noisy1spps.resize(3 * w * h);

	LOG("  Loading data of frame %d\n", frame);

    Operation_result result = load_image(frameInputData.albedos.data(), ALBEDO_FILE_NAME, frame);
    if(!result.success)
    {
        LOG("Albedo buffer loading failed, reason: %s (frame %d)\n", result.error_message.c_str(), frame);
        return false;
    }

    result = load_image(frameInputData.normals.data(), NORMAL_FILE_NAME, frame);
    if(!result.success)
    {
        LOG("Normal buffer loading failed, reason: %s (frame %d)\n", result.error_message.c_str(), frame);
        return false;
    }

    result = load_image(frameInputData.positions.data(), POSITION_FILE_NAME, frame);
    if(!result.success)
    {
        LOG("Position buffer loading failed, reason: %s (frame %d)\n", result.error_message.c_str(), frame);
        return false;
    }

    result = load_image(frameInputData.noisy1spps.data(), NOISY_FILE_NAME, frame);
    if(!result.success)
    {
        LOG("Position buffer loading failed, reason: %s (frame %d)\n", result.error_message.c_str(), frame);
        return false;
    }
            
	return true;
}
