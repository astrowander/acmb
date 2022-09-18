#include "imageencoder.h"
#include <fstream>
#include <filesystem>
#include "PPM/ppmencoder.h"
#include "Tiff/TiffEncoder.h"

ACMB_NAMESPACE_BEGIN

void ImageEncoder::Attach(std::shared_ptr<std::ostream> pStream)
{
    if (!pStream)
        throw std::invalid_argument("pStream");

    _pStream = pStream;
}

void ImageEncoder::Attach(const std::string &fileName)
{
    std::shared_ptr<std::ofstream> pStream(new std::ofstream(fileName, std::ios_base::binary | std::ios_base::out));
    if (!pStream->is_open())
        throw std::invalid_argument("pStream");

    Attach(pStream);
}

void ImageEncoder::Detach()
{
    _pStream.reset();
}

std::shared_ptr<ImageEncoder> ImageEncoder::Create(const std::string &fileName)
{
    auto path = std::filesystem::path(fileName);
    auto extension = path.extension().string();
    std::shared_ptr<ImageEncoder> pEncoder;
    if (PpmEncoder::GetExtensions().contains(extension))
    {
        pEncoder.reset(new PpmEncoder(PpmMode::Binary));
    }
    else if ( TiffEncoder::GetExtensions().contains( extension ) )
    {
        pEncoder.reset( new TiffEncoder() );
    }

    if (!pEncoder)
        throw std::invalid_argument("fileName");

    return pEncoder;
}

ACMB_NAMESPACE_END
