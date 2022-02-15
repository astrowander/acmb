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

Registrator::Registrator(double threshold, uint32_t minStarSize, uint32_t maxStarSize)
: _threshold(threshold)
, _minStarSize(minStarSize)
, _maxStarSize(maxStarSize)
{

}

void Registrator::Registrate(std::shared_ptr<IBitmap> pBitmap)
{
    _stars.clear();
    _centralStars.clear();

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
    Rect centralRoi{ 2 * w / 5, 2 * h / 5, w / 5, h / 5 };
    
    if (BytesPerChannel(pBitmap->GetPixelFormat()) == 1)
    {
        _stars = Registrate<PixelFormat::Gray8>(roi);
    }
    else
    {
        _stars = Registrate<PixelFormat::Gray16>(roi);
    }

    SortStars(_stars);

    for (const auto& star : _stars)
    {
        if (star.rect.Overlaps(centralRoi))
            _centralStars.push_back(star);
    }
}

std::vector<Star> Registrator::GetStars() const 
{
    return _stars;
}

std::vector<Star> Registrator::GetCentralStars() const 
{
    return _centralStars;
}
