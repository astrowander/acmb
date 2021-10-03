#include "imagedecoder.h"
#include <fstream>

void ImageDecoder::Attach(std::unique_ptr<std::istream> pStream)
{
    if (!pStream)
        throw std::runtime_error("pStream is null");

    _pStream = std::move(pStream);
}

void ImageDecoder::Attach(const std::string &fileName)
{
    std::unique_ptr<std::ifstream> pStream(new std::ifstream(fileName));
    if (!pStream->is_open())
        throw std::invalid_argument("fileName");

    Attach(std::move(pStream));
}

std::unique_ptr<std::istream> ImageDecoder::Detach()
{
    _pStream->seekg(0);
    return std::move(_pStream);
}

std::unique_ptr<std::istringstream> ImageDecoder::ReadLine()
{
    std::string res;
    std::getline(*_pStream, res);
    return std::make_unique<std::istringstream>(res);
}
