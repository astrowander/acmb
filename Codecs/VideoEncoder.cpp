#include "VideoEncoder.h"
#include "../Core/bitmap.h"

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

ACMB_NAMESPACE_BEGIN

void VideoEncoder::BitmapToYuv( std::shared_ptr<IBitmap> pBitmap )
{
    if ( !pBitmap )
        throw std::invalid_argument( "pBitmap" );

    if ( pBitmap->GetPixelFormat() != PixelFormat::RGB24 )
        throw std::invalid_argument( "unsupported pixel format" );

    if ( pBitmap->GetWidth() != _width || pBitmap->GetHeight() != _height )
        throw std::invalid_argument( "image size mismatch" );

    constexpr int channelCount = 3;
    const int lumaSize = _width * _height;
    const int chromaSize = _width * _height / 4;

    uint8_t* pFrame = _yuv.data();

    tbb::parallel_for( tbb::blocked_range<int>( 0, _height ), [&] ( const tbb::blocked_range<int>& range )
    {
        for ( int line = range.begin(); line < range.end(); ++line )
            //for ( int line = 0; line < height; ++line )
        {
            auto pScanline = ( uint8_t* ) pBitmap->GetPlanarScanline( line );
            int lPos = line * _width;
            int uPos = line * _width / 4;
            int vPos = line * _width / 4;

            if ( !(line % 2) )
            {
                for ( size_t x = 0; x < _width; x += 2 )
                {
                    uint8_t r = pScanline[x * channelCount];
                    uint8_t g = pScanline[x * channelCount + 1];
                    uint8_t b = pScanline[x * channelCount + 2];

                    pFrame[lPos++] = ((66 * r + 129 * g + 25 * b) >> 8) + 16;
                    pFrame[lumaSize + uPos++] = ((-38 * r + -74 * g + 112 * b) >> 8) + 128;
                    pFrame[lumaSize + chromaSize + vPos++] = ((112 * r + -94 * g + -18 * b) >> 8) + 128;

                    r = pScanline[(x + 1) * channelCount];
                    g = pScanline[(x + 1) * channelCount + 1];
                    b = pScanline[(x + 1) * channelCount + 2];

                    pFrame[lPos++] = ((66 * r + 129 * g + 25 * b) >> 8) + 16;
                }
            }
            else
            {
                for ( size_t x = 0; x < _width; x += 1 )
                {
                    uint8_t r = pScanline[x * channelCount];
                    uint8_t g = pScanline[x * channelCount + 1];
                    uint8_t b = pScanline[x * channelCount + 2];
                    pFrame[lPos++] = ((66 * r + 129 * g + 25 * b) >> 8) + 16;
                }
            }
        }
    } );
}

void VideoEncoder::SetFrameRate( uint32_t rate )
{
    _frameRate = rate;
}

uint32_t VideoEncoder::GetFrameRate() const
{
    return _frameRate;
}

void VideoEncoder::SetTotalFrames( uint32_t frames )
{
    _totalFrames = frames;
}

uint32_t VideoEncoder::GetTotalFrames() const
{
    return _totalFrames;
}

ACMB_NAMESPACE_END