#include "imagedecoder.h"
#include <fstream>

void ImageDecoder::Attach(std::shared_ptr<std::istream> pStream)
{
    if (!pStream)
        throw std::runtime_error("pStream is null");

    _pStream = pStream;
}

void ImageDecoder::Attach(const std::string &fileName)
{
    std::unique_ptr<std::ifstream> pStream(new std::ifstream(fileName));
    if (!pStream->is_open())
        throw std::invalid_argument("fileName");

    Attach(std::move(pStream));
}

void ImageDecoder::Detach()
{
    _pStream.reset();
}

std::unique_ptr<std::istringstream> ImageDecoder::ReadLine()
{
    std::string res;
    std::getline(*_pStream, res);
    return std::make_unique<std::istringstream>(res);
}
