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

    constexpr int N = 12;
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

template<PixelFormat pixelFormat>
std::vector<PointD> DetectFeaturesImpl( std::shared_ptr<Bitmap<pixelFormat>> pBitmap, float threshold, int minChannel )
{
    constexpr PixelFormat grayFormat = ConstructPixelFormat( BitsPerChannel( pixelFormat ), 1 );
    auto pGrayBitmap = std::static_pointer_cast< Bitmap<grayFormat>>(pBitmap);

    struct Cluster
    {
        PointD center;
        int size;
    };

    std::vector<Cluster> c;

    for ( int y = 3; y < int( pBitmap->GetHeight() ) - 3; ++y )
    {
        auto pPixel = pGrayBitmap->GetScanline( y );
        for ( int x = 3; x < int( pBitmap->GetWidth() ) - 3; ++x )
        {
            if ( CheckPixel( &pPixel[x], pBitmap->GetWidth(), threshold, minChannel ) )
            {
                bool found = false;
                for ( size_t i = 0; i < c.size(); ++i )
                {
                    auto& center = c[i].center;
                    if ( fabs( center.x - x ) < 3 && fabs( center.y - y ) < 3 )
                    {
                        center.x = (center.x * c[i].size + x) / (c[i].size + 1);
                        center.y = (center.y * c[i].size + y) / (c[i].size + 1);
                        ++c[i].size;
                        found = true;
                        break;
                    }
                }

                if ( !found )
                {
                    Cluster cluster { .center = PointD{ double(x), double(y) }, .size = 1 };
                    c.push_back( cluster );
                }
            }
            //result.push_back( { x, y } );
        }
    }

    std::sort( c.begin(), c.end(), []( const Cluster& c1, const Cluster& c2 ) 
    { 
        if ( c1.size == c2.size )
            return c1.center.y == c2.center.y ? c1.center.x < c2.center.x : c1.center.y < c2.center.y;

        return c1.size > c2.size; 
    } );

    std::vector<PointD> result( 30 );

    for ( int i = 0; i < result.size(); ++i )
    {
        result[i] = c[i].center;
    }

    return result;
}

std::vector<PointD> DetectFeatures( IBitmapPtr pBitmap, float threshold, int minChannel )
{
    if ( GetColorSpace( pBitmap->GetPixelFormat() ) != ColorSpace::Gray )
        throw std::invalid_argument( "FAST detector only supports grayscale images." );



    if ( pBitmap->GetPixelFormat() == PixelFormat::Gray8 )
    {
        return DetectFeaturesImpl<PixelFormat::Gray8>( std::static_pointer_cast<Bitmap<PixelFormat::Gray8>>(pBitmap), threshold, minChannel );
    }
    else //if ( pBitmap->GetPixelFormat() == PixelFormat::Gray16 )
    {
        return DetectFeaturesImpl<PixelFormat::Gray16>( std::static_pointer_cast< Bitmap<PixelFormat::Gray16> >(pBitmap), threshold, minChannel );
    }
}

ACMB_NAMESPACE_END