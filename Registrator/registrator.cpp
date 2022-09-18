#include "registrator.h"
#include "./../Transforms/converter.h"
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

ACMB_NAMESPACE_BEGIN

void SortStars(std::vector<Star>& stars)
{
    if (!stars.empty())
    {
        std::sort(stars.begin(), stars.end(), [](auto& a, auto& b) {return a.luminance > b.luminance; });
        auto maxLuminance = stars[0].luminance;

        for (auto& star : stars)
        {
            star.luminance /= maxLuminance;
        }
    }
}

Registrator::Registrator( double threshold, uint32_t _minStarSize, uint32_t _maxStarSize )
:_threshold( threshold )
, _minStarSize( _minStarSize )
, _maxStarSize( _maxStarSize )
{

}

void Registrator::Registrate(std::shared_ptr<IBitmap> pBitmap)
{
    if ( !pBitmap )
        throw std::invalid_argument( "pBitmap is null" );

    auto [hTileCount, vTileCount] = GetTileCounts( pBitmap->GetWidth(), pBitmap->GetHeight() );
    
    _stars.clear();
    _stars.resize(hTileCount * vTileCount);

    if (GetColorSpace(pBitmap->GetPixelFormat()) == ColorSpace::Gray)
    {
        _pBitmap = pBitmap;
    }
    else
    {
        auto pConverter = Converter::Create(pBitmap, BytesPerChannel(pBitmap->GetPixelFormat()) == 1 ? PixelFormat::Gray8 : PixelFormat::Gray16);
        _pBitmap = pConverter->RunAndGetBitmap();
    }    

    oneapi::tbb::parallel_for( oneapi::tbb::blocked_range<int>( 0, hTileCount * vTileCount ), [this] ( const oneapi::tbb::blocked_range<int>& range )
    {
        const auto w = tileSize;
        const auto h = tileSize;

        auto [hTileCount, vTileCount] = GetTileCounts( _pBitmap->GetWidth(), _pBitmap->GetHeight() );

        for ( int i = range.begin(); i < range.end(); ++i )
        {
            const auto y = i / hTileCount;
            const auto x = i % hTileCount;

            Rect roi{ int( x * w ), int( y * h ), int( ( x < hTileCount - 1 ) ? w : _pBitmap->GetWidth() - x * w ), int( ( y < vTileCount - 1 ) ? h : _pBitmap->GetHeight() - y * h ) };

            std::vector<Star> tileStars;
            if ( BytesPerChannel( _pBitmap->GetPixelFormat() ) == 1 )
            {
                tileStars = Registrate<PixelFormat::Gray8>( roi );
            }
            else
            {
                tileStars = Registrate<PixelFormat::Gray16>( roi );
            }

            SortStars( tileStars );

            _stars[i] = tileStars;
        }
    } );
}

const std::vector<std::vector<Star>>& Registrator::GetStars() const
{
    return _stars;
}

ACMB_NAMESPACE_END