#include "FastDetector.h"

ACMB_NAMESPACE_BEGIN

template<typename ChannelType>
bool CheckPixel( ChannelType* pPixel, int width, float threshold, int minChannel )
{
    enum class PixelState
    {
        Similar,
        Darker,        
        Brighter
    };

    ChannelType pixelVal = *pPixel;
    if ( pixelVal < minChannel )
        return false;

    auto checkNeighbour = [&] ( ChannelType neighbourVal )
    {
        if ( neighbourVal > ( 1.0f + threshold ) * pixelVal )
            return PixelState::Darker;

        if ( neighbourVal < ( 1.0f - threshold ) * pixelVal )
            return PixelState::Brighter;

        return PixelState::Similar;
    };

    PixelState neighbours[16] =
    {
        checkNeighbour( pPixel[-3 * width -1] ),
        checkNeighbour( pPixel[-3 * width] ),
        checkNeighbour( pPixel[-3 * width +1] ),
        checkNeighbour( pPixel[-2 * width + 2] ),
        checkNeighbour( pPixel[-width + 3] ),
        checkNeighbour( pPixel[3] ),
        checkNeighbour( pPixel[width + 3] ),
        checkNeighbour( pPixel[2 * width + 2] ),
        checkNeighbour( pPixel[3 * width + 1] ),
        checkNeighbour( pPixel[3 * width] ),
        checkNeighbour( pPixel[3 * width -1] ),
        checkNeighbour( pPixel[2 * width -2] ),
        checkNeighbour( pPixel[width - 3] ),
        checkNeighbour( pPixel[-3] ),
        checkNeighbour( pPixel[-width - 3] ),
        checkNeighbour( pPixel[-2 * width -2] )
    };

    constexpr int N = 8;
    constexpr int halfN = N / 2;
    for ( int i = halfN; i < 16 + halfN; ++i )
    {
        if ( neighbours[i] == PixelState::Similar )
        {
            i += halfN - 1;
            continue;
        }

        bool res = true;

        for ( int j = i - halfN; j < i - halfN + N; ++j )
        {
            if ( neighbours[j] != neighbours[i] )
            {
                res = false;
                break;
            }
        }

        if ( res )
            return true;
    }

    return false;
    
}

std::vector<Point> DetectFeatures( IBitmapPtr pBitmap, float threshold, int minChannel )
{
    if ( GetColorSpace( pBitmap->GetPixelFormat() ) != ColorSpace::Gray )
        throw std::invalid_argument( "FAST detector only supports grayscale images." );

    std::vector<Point> result;

    if ( pBitmap->GetPixelFormat() == PixelFormat::Gray8 )
    {
        auto pGrayBitmap = std::static_pointer_cast< Bitmap<PixelFormat::Gray8> >(pBitmap);
        for ( int y = 3; y < int( pBitmap->GetHeight() ) - 3; ++y )
        {
            auto pPixel = pGrayBitmap->GetScanline( y );
            for ( int x = 3; x < int( pBitmap->GetWidth() ) - 3; ++x )
            {
                if ( CheckPixel( &pPixel[x], pBitmap->GetWidth(), threshold, minChannel ) )
                    result.push_back( { x, y } );
            }
        }

        return result;
    }

    //if ( pBitmap->GetPixelFormat() == PixelFormat::Gray16 )
    
    auto pGrayBitmap = std::static_pointer_cast< Bitmap<PixelFormat::Gray16> >(pBitmap);
    for ( int y = 3; y < int( pBitmap->GetHeight() ) - 3; ++y )
    {
        auto pPixel = pGrayBitmap->GetScanline( y );
        for ( int x = 3; x < int( pBitmap->GetWidth() ) - 3; ++x )
        {
            if ( CheckPixel( &pPixel[x], pBitmap->GetWidth(), threshold, minChannel ) )
                result.push_back( { x, y } );
        }
    }    
    
    return result;
}

ACMB_NAMESPACE_END