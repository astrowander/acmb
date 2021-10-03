#include "imageencoder.h"
#include <fstream>

void ImageEncoder::Attach(std::unique_ptr<std::ostream> pStream)
{
    if (!pStream)
        throw std::invalid_argument("pStream");

    _pStream = std::move(pStream);
}

void ImageEncoder::Attach(const std::string &fileName)
{
    std::unique_ptr<std::ofstream> pStream(new std::ofstream(fileName));
    if (!pStream->is_open())
        throw std::invalid_argument("pStream");

    Attach(std::move(pStream));
}

std::unique_ptr<std::ostream> ImageEncoder::Detach()
{
    return std::move(_pStream);
}
