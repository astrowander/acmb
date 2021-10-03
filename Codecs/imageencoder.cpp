#include "imageencoder.h"
#include <fstream>

void ImageEncoder::Attach(std::shared_ptr<std::ostream> pStream)
{
    if (!pStream)
        throw std::invalid_argument("pStream");

    _pStream = pStream;
}

void ImageEncoder::Attach(const std::string &fileName)
{
    std::shared_ptr<std::ofstream> pStream(new std::ofstream(fileName));
    if (!pStream->is_open())
        throw std::invalid_argument("pStream");

    Attach(pStream);
}

void ImageEncoder::Detach()
{
    _pStream.reset();
}
