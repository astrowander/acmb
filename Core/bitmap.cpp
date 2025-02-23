#include "bitmap.h"
#include <fstream>
#include <filesystem>
#include "../Codecs/imageencoder.h"
#include "../Codecs/imagedecoder.h"

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>


ACMB_NAMESPACE_BEGIN

std::string GetDirectory( const std::string& fileName )
{
    size_t pos = fileName.find_last_of( "\\/" );
    return ( std::string::npos == pos ) ? "" : fileName.substr( 0, pos );
}

std::shared_ptr<IBitmap> IBitmap::Create(std::shared_ptr<std::istream> pStream, PixelFormat outputFormat )
{
    auto pDecoder = ImageDecoder::Create( pStream, outputFormat );
    const auto res =  pDecoder->ReadBitmap();
    pDecoder->Detach();
    return res;
}

std::shared_ptr<IBitmap> IBitmap::Create(const std::string &fileName, PixelFormat outputFormat )
{
    auto pDecoder = ImageDecoder::Create( fileName, outputFormat );
    const auto res = pDecoder->ReadBitmap();
    pDecoder->Detach();
    return res;
}

std::shared_ptr<IBitmap> IBitmap::Create(uint32_t width, uint32_t height, PixelFormat pixelFormat)
{
    switch (pixelFormat)
    {
    case PixelFormat::Gray8:
        return std::make_shared<Bitmap<PixelFormat::Gray8>>(width, height);
    case PixelFormat::Gray16:
        return std::make_shared<Bitmap<PixelFormat::Gray16>>(width, height);
    case PixelFormat::YUV24:
        return std::make_shared<Bitmap<PixelFormat::YUV24>>( width, height );
    case PixelFormat::RGB24:
        return std::make_shared<Bitmap<PixelFormat::RGB24>>(width, height);
    case PixelFormat::RGB48:
        return std::make_shared<Bitmap<PixelFormat::RGB48>>(width, height);
    case PixelFormat::Bayer16:
        return std::make_shared<Bitmap<PixelFormat::Bayer16>>( width, height );
    case PixelFormat::RGBA32:
        return std::make_shared<Bitmap<PixelFormat::RGBA32>>( width, height );
    case PixelFormat::RGBA64:
        return std::make_shared<Bitmap<PixelFormat::RGBA64>>( width, height );
    default:
        throw std::runtime_error("not implemented");
    }
}

std::shared_ptr<IBitmap> IBitmap::Create( uint32_t width, uint32_t height, IColorPtr pColor )
{
    const auto pixelFormat = pColor->GetPixelFormat();
    switch (pixelFormat)
    {
    case PixelFormat::Gray8:
        return std::make_shared<Bitmap<PixelFormat::Gray8>>(width, height, std::static_pointer_cast<Color<PixelFormat::Gray8>>(pColor));
    case PixelFormat::Gray16:
        return std::make_shared<Bitmap<PixelFormat::Gray16>>(width, height, std::static_pointer_cast< Color<PixelFormat::Gray16> >(pColor) );
    case PixelFormat::RGB24:
        return std::make_shared<Bitmap<PixelFormat::RGB24>>(width, height, std::static_pointer_cast< Color<PixelFormat::RGB24> >(pColor) );
    case PixelFormat::YUV24:
        return std::make_shared<Bitmap<PixelFormat::YUV24>>( width, height, std::static_pointer_cast< Color<PixelFormat::YUV24> >(pColor) );
    case PixelFormat::RGB48:
        return std::make_shared<Bitmap<PixelFormat::RGB48>>(width, height, std::static_pointer_cast< Color<PixelFormat::RGB48> >(pColor) );
    case PixelFormat::RGBA32:
        return std::make_shared<Bitmap<PixelFormat::RGBA32>>( width, height, std::static_pointer_cast< Color<PixelFormat::RGBA32> >(pColor) );
    case PixelFormat::RGBA64:
        return std::make_shared<Bitmap<PixelFormat::RGBA64>>( width, height, std::static_pointer_cast< Color<PixelFormat::RGBA64> >(pColor) );
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

template<PixelFormat pixelFormat>
void Bitmap<pixelFormat>::Fill( std::shared_ptr<Color<pixelFormat>> pColor )
{
    if ( !pColor )
        throw std::invalid_argument( "pColor" );

    ChannelType channels[channelCount];
    for ( uint32_t i = 0; i < channelCount; ++i )
    {
        channels[i] = ChannelType( pColor->GetChannel( i ) );
    }

    oneapi::tbb::parallel_for( oneapi::tbb::blocked_range<uint32_t>( 0, _height ), [&] ( const oneapi::tbb::blocked_range<uint32_t>& r )
    {
        for ( uint32_t i = r.begin(); i < r.end(); ++i )
        {
            auto pScanline = GetScanline( i );
            for ( uint32_t i = 0; i < _width; ++i )
            {
                for ( uint32_t ch = 0; ch < ChannelCount( pixelFormat ); ++ch )
                {
                    *pScanline++ = ChannelType( pColor->GetChannel( ch ) );
                }
            }
        }
    } );
}

ACMB_NAMESPACE_END
