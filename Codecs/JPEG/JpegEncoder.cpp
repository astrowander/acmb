#include "JpegEncoder.h"
#include "toojpeg/toojpeg.h"
#include "../../Core/bitmap.h"

void WriteByte( uint8_t byte )
{

}

JpegEncoder::JpegEncoder( uint8_t quality, bool downsample )
: _quality( quality )
, _downsample( downsample )
{

}

void JpegEncoder::WriteBitmap( std::shared_ptr<IBitmap> pBitmap )
{
    if ( !pBitmap )
        throw std::invalid_argument( "pBitmap is null" );

    if ( BytesPerChannel( pBitmap->GetPixelFormat() ) != 1 )
        throw std::invalid_argument( "JpegEncoder does not support extended pixel formats" );

    auto writeByte = [this] ( uint8_t byte )
    {
        _pStream->put( char( byte ) );
    };

    if ( !TooJpeg::writeJpeg( writeByte, pBitmap->GetPlanarScanline(0), pBitmap->GetWidth(), pBitmap->GetHeight(), GetColorSpace(pBitmap->GetPixelFormat()) == ColorSpace::RGB, _quality, _downsample ) )
         throw std::runtime_error("unable to write jpeg");
}

uint8_t JpegEncoder::GetQuality()
{
    return _quality;
}

bool JpegEncoder::GetDownsample()
{
    return _downsample;
}

void JpegEncoder::SetQuality( uint8_t val )
{
    _quality = val;
}

void JpegEncoder::SetDownsample( bool val )
{
    _downsample = val;
}
