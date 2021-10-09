#include "imagedecoder.h"
#include <fstream>
#include <filesystem>
#include "PPM/ppmdecoder.h"

void ImageDecoder::Attach(std::shared_ptr<std::istream> pStream)
{
    if (!pStream)
        throw std::runtime_error("pStream is null");

    _pStream = pStream;
}

void ImageDecoder::Attach(const std::string &fileName)
{
    std::shared_ptr<std::ifstream> pStream(new std::ifstream(fileName));
    if (!pStream->is_open())
        throw std::invalid_argument("fileName");

    Attach(pStream);
}

void ImageDecoder::Detach()
{
    _pStream.reset();
}

std::shared_ptr<ImageDecoder> ImageDecoder::Create(const std::string &fileName)
{
    auto path = std::filesystem::path(fileName);
    auto extension = path.extension();
    std::shared_ptr<ImageDecoder> pDecoder;
    if (extension == ".pgm" || extension == ".ppm")
    {
        pDecoder.reset(new PpmDecoder());
    }

    if (!pDecoder)
        throw std::invalid_argument("fileName");

    return pDecoder;
}

std::unique_ptr<std::istringstream> ImageDecoder::ReadLine()
{
    std::string res;
    std::getline(*_pStream, res);
    return std::make_unique<std::istringstream>(res);
}
