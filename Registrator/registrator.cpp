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

Registrator::Registrator(uint32_t hTileCount, uint32_t vTileCount, double threshold, uint32_t minStarSize, uint32_t maxStarSize)
: _hTileCount(hTileCount)
, _vTileCount(vTileCount)
, _threshold(threshold)
, _minStarSize(minStarSize)
, _maxStarSize(maxStarSize)
{

}

void Registrator::Registrate(std::shared_ptr<IBitmap> pBitmap)
{
    _stars.clear();
    _stars.reserve(_hTileCount * _vTileCount);

    if (GetColorSpace(pBitmap->GetPixelFormat()) == ColorSpace::Gray)
    {
        _pBitmap = pBitmap;
    }
    else
    {
        _pBitmap = Convert(pBitmap, BytesPerChannel(pBitmap->GetPixelFormat()) == 1 ? PixelFormat::Gray8 : PixelFormat::Gray16);
    }
    
    const auto w = pBitmap->GetWidth() / _hTileCount;
    const auto h = pBitmap->GetHeight() / _vTileCount;

    for (uint32_t y = 0; y < _vTileCount; ++y)
    for (uint32_t x = 0; x < _hTileCount; ++x)
    {
        Rect roi{ x * w, y * h, (x < _hTileCount - 1) ? w : pBitmap->GetWidth() - x * w, (y < _vTileCount - 1) ? h : pBitmap->GetHeight() - y * h };

        if (BytesPerChannel(pBitmap->GetPixelFormat()) == 1)
        {
            _stars.push_back(Registrate<PixelFormat::Gray8>(roi));
        }
        else
        {
            _stars.push_back(Registrate<PixelFormat::Gray16>(roi));
        }

        SortStars(_stars.back());
    }
}

const std::vector<std::vector<Star>>& Registrator::GetStars() const
{
    return _stars;
}
