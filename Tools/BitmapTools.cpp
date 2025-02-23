#include "BitmapTools.h"

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

ACMB_NAMESPACE_BEGIN

std::shared_ptr<Bitmap<PixelFormat::YUV24>> YUVBitmapFromPlanarData( std::vector<uint8_t>& yuv, uint32_t width, uint32_t height )
{
    if ( width & 1 || height & 1 )
        throw std::invalid_argument( "width and height should be even" );

    const size_t lumaSize = width * height;
    const size_t chromaSize = lumaSize / 4;

    if ( yuv.size() != lumaSize + chromaSize * 2 )
        throw std::invalid_argument( "yuv size is invalid" );

    auto pBitmap = std::make_shared<Bitmap<PixelFormat::YUV24>>( width, height );

    const uint8_t* yPlane = yuv.data();
    const uint8_t* uPlane = yPlane + lumaSize;
    const uint8_t* vPlane = uPlane + chromaSize;

    tbb::parallel_for( tbb::blocked_range<uint32_t>( 0, height ), [&]( const tbb::blocked_range<uint32_t>& r )    
    {
        for ( uint32_t line = r.begin(); line < r.end(); line++ )        
        {
            auto pPixel = pBitmap->GetScanline( line );
            const size_t lPos = line * width;
            const size_t uPos = (line / 2) * (width / 2);
            const size_t vPos = (line / 2) * (width / 2);

            for ( uint32_t x = 0; x < width; x++ )
            {
                pPixel[0] = yPlane[lPos + x];
                pPixel[1] = uPlane[uPos + x / 2];
                pPixel[2] = vPlane[vPos + x / 2];
                pPixel += 3;
            }
        }
    } );

    return pBitmap;
}

void PlanarDataFromYUVBitmap( std::shared_ptr<Bitmap<PixelFormat::YUV24>> pBitmap, std::vector<uint8_t>& yuv )
{
    if ( !pBitmap )
        throw std::invalid_argument( "pBitmap is null" );

    const auto width = pBitmap->GetWidth();
    const auto height = pBitmap->GetHeight();
    
    if ( width & 1 || height & 1 )
        throw std::invalid_argument( "width and height should be even" );

    constexpr size_t channelCount = 3;
    const size_t lumaSize = width * height;
    const size_t chromaSize = lumaSize / 4;

    if ( yuv.size() != lumaSize + chromaSize * 2 )
        throw std::invalid_argument( "yuv size is invalid" );

    uint8_t* yPlane = yuv.data();
    uint8_t* uPlane = yPlane + lumaSize;
    uint8_t* vPlane = uPlane + chromaSize;

    tbb::parallel_for( tbb::blocked_range<uint32_t>( 0, height / 2 ), [&] ( const tbb::blocked_range<uint32_t>& r )
    {
        for ( uint32_t linePair = r.begin(); linePair < r.end(); ++linePair )
        {
            const auto pFirstLine = pBitmap->GetScanline( linePair * 2 );
            const auto pSecondLine = pBitmap->GetScanline( linePair * 2 + 1 );

            uint8_t* yFirstLine = yPlane + linePair * 2 * width;
            uint8_t* ySecondLine = yFirstLine + width;
            uint8_t* uLine = uPlane + linePair * width / 2;
            uint8_t* vLine = vPlane + linePair * width / 2;

            for ( uint32_t columnPair = 0; columnPair < width / 2; ++columnPair )
            {
                const uint8_t* pUpperLeftPixel = pFirstLine + columnPair * 2 * channelCount;
                const uint8_t* pUpperRightPixel = pUpperLeftPixel + channelCount;
                const uint8_t* pBottomLeftPixel = pSecondLine + columnPair * 2 * channelCount;
                const uint8_t* pBottomRightPixel = pBottomLeftPixel + channelCount;

                yFirstLine[columnPair * 2] = pUpperLeftPixel[0];
                yFirstLine[columnPair * 2 + 1] = pUpperRightPixel[0];
                ySecondLine[columnPair * 2] = pBottomLeftPixel[0];
                ySecondLine[columnPair * 2 + 1] = pBottomRightPixel[0];                

                uLine[columnPair] = (pUpperLeftPixel[1] + pUpperRightPixel[1] + pBottomLeftPixel[1] + pBottomRightPixel[1]) / 4;
                vLine[columnPair] = (pUpperLeftPixel[2] + pUpperRightPixel[2] + pBottomLeftPixel[2] + pBottomRightPixel[2]) / 4;
            }
        }
    } );
}

ACMB_NAMESPACE_END