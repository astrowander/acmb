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
: IParallel(hTileCount * vTileCount)
, _hTileCount(hTileCount)
, _vTileCount(vTileCount)
, _threshold(threshold)
, _minStarSize(minStarSize)
, _maxStarSize(maxStarSize)
{

}

void Registrator::Registrate(std::shared_ptr<IBitmap> pBitmap)
{
    _stars.clear();
    _stars.resize(_hTileCount * _vTileCount);

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
    const auto w = _pBitmap->GetWidth() / _hTileCount;
    const auto h = _pBitmap->GetHeight() / _vTileCount;

    const auto y = i / _hTileCount;
    const auto x = i % _hTileCount;

    Rect roi{ x * w, y * h, (x < _hTileCount - 1) ? w : _pBitmap->GetWidth() - x * w, (y < _vTileCount - 1) ? h : _pBitmap->GetHeight() - y * h };

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
