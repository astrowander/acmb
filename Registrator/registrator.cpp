#include "registrator.h"
#include "./../Transforms/converter.h"

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
    auto [hTileCount, vTileCount] = GetTileCounts( pBitmap->GetWidth(), pBitmap->GetHeight() );
    SetJobCount( hTileCount * vTileCount );

    _stars.clear();
    _stars.resize(hTileCount * vTileCount);

    if (GetColorSpace(pBitmap->GetPixelFormat()) == ColorSpace::Gray)
    {
        _pBitmap = pBitmap;
    }
    else
    {
        auto pConverter = BaseConverter::Create(pBitmap, BytesPerChannel(pBitmap->GetPixelFormat()) == 1 ? PixelFormat::Gray8 : PixelFormat::Gray16);
        _pBitmap = pConverter->RunAndGetBitmap();
    }    

    DoParallelJobs();
}

const std::vector<std::vector<Star>>& Registrator::GetStars() const
{
    return _stars;
}

void Registrator::Job(uint32_t i)
{
    const auto w = tileSize;
    const auto h = tileSize;

    auto [hTileCount, vTileCount] = GetTileCounts( _pBitmap->GetWidth(), _pBitmap->GetHeight() );

    const auto y = i / hTileCount;
    const auto x = i % hTileCount;

    Rect roi{ x * w, y * h, (x < hTileCount - 1) ? w : _pBitmap->GetWidth() - x * w, (y < vTileCount - 1) ? h : _pBitmap->GetHeight() - y * h };

    std::vector<Star> tileStars;
    if (BytesPerChannel(_pBitmap->GetPixelFormat()) == 1)
    {
        tileStars = Registrate<PixelFormat::Gray8>(roi);
    }
    else
    {
        tileStars = Registrate<PixelFormat::Gray16>(roi);
    }

    SortStars(tileStars);

    _mutex.lock();
    _stars[i] = tileStars;
    _mutex.unlock();    
}
