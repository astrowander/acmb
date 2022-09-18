#include "bitmap.h"
#include <fstream>
#include <filesystem>
#include "../Codecs/imagedecoder.h"
#include "../Codecs/imageencoder.h"

ACMB_NAMESPACE_BEGIN

std::string GetDirectory( const std::string& fileName )
{
    size_t pos = fileName.find_last_of( "\\/" );
    return ( std::string::npos == pos ) ? "" : fileName.substr( 0, pos );
}

std::shared_ptr<IBitmap> IBitmap::Create(const std::string &fileName)
{
    auto pDecoder = ImageDecoder::Create(fileName);
    pDecoder->Attach(fileName);
    return pDecoder->ReadBitmap();
}

std::shared_ptr<IBitmap> IBitmap::Create(uint32_t width, uint32_t height, PixelFormat pixelFormat)
{
    switch (pixelFormat)
    {
    case PixelFormat::Gray8:
        return std::make_shared<Bitmap<PixelFormat::Gray8>>(width, height);
    case PixelFormat::Gray16:
        return std::make_shared<Bitmap<PixelFormat::Gray16>>(width, height);
    case PixelFormat::RGB24:
        return std::make_shared<Bitmap<PixelFormat::RGB24>>(width, height);
    case PixelFormat::RGB48:
        return std::make_shared<Bitmap<PixelFormat::RGB48>>(width, height);
    default:
        throw std::runtime_error("not implemented");

    }
}

void IBitmap::Save(std::shared_ptr<IBitmap> pBitmap, const std::string &fileName)
{
    if (!pBitmap)
        throw std::invalid_argument("pBitmap is null");

    auto dir = GetDirectory( fileName );
    if ( !std::filesystem::exists( dir ) )
        std::filesystem::create_directory( dir );

    auto pEncoder = ImageEncoder::Create(fileName);
    pEncoder->Attach(fileName);
    return pEncoder->WriteBitmap(pBitmap);
}

ACMB_NAMESPACE_END