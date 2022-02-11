#include "registrator.h"
#include "./../Transforms/converter.h"

Registrator::Registrator(double threshold, uint32_t minStarSize, uint32_t maxStarSize)
: _threshold(threshold)
, _minStarSize(minStarSize)
, _maxStarSize(maxStarSize)
{

}

std::vector<Star> Registrator::Registrate(std::shared_ptr<IBitmap> pBitmap)
{
    if (GetColorSpace(pBitmap->GetPixelFormat()) == ColorSpace::Gray)
    {
        _pBitmap = pBitmap;
    }
    else
    {
        _pBitmap = Convert(pBitmap, BytesPerChannel(pBitmap->GetPixelFormat()) == 1 ? PixelFormat::Gray8 : PixelFormat::Gray16);
    }
    
    const auto w = pBitmap->GetWidth();
    const auto h = pBitmap->GetHeight();
    

    Rect roi{ 0, 0, w, h };
    
    std::vector<Star> res;
    if (BytesPerChannel(pBitmap->GetPixelFormat()) == 1)
        res = Registrate<PixelFormat::Gray8>(roi);
    else
        res = Registrate<PixelFormat::Gray16>(roi);

    if (!res.empty())
    {
        std::sort(res.begin(), res.end(), [](auto& a, auto& b) {return a.luminance > b.luminance; });
        auto maxLuminance = res[0].luminance;

        for (auto& star : res)
        {
            star.luminance /= maxLuminance;
        }
    }

    return res;
}
