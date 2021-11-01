#include "registrator.h"
#include "./../Transforms/converter.h"

Registrator::Registrator(double threshold, uint32_t minStarSize, uint32_t maxStarSize)
: _threshold(threshold)
, _minStarSize(minStarSize)
, _maxStarSize(maxStarSize)
{

}

std::shared_ptr<AlignmentDataset> Registrator::Registrate(std::shared_ptr<IBitmap> pBitmap)
{
    if (GetColorSpace(pBitmap->GetPixelFormat()) == ColorSpace::Gray)
    {
        _pBitmap = pBitmap;
    }
    else
    {
        _pBitmap = Convert(pBitmap, BytesPerChannel(pBitmap->GetPixelFormat()) == 1 ? PixelFormat::Gray8 : PixelFormat::Gray16);
    }
    
    _dataset.reset(new AlignmentDataset());

    if (BytesPerChannel(pBitmap->GetPixelFormat()) == 1)
        Registrate<PixelFormat::Gray8>();
    else
        Registrate<PixelFormat::Gray16>();

    std::sort(_dataset->stars.begin(), _dataset->stars.end(), [](auto& a, auto& b) {return a.luminance > b.luminance; });
    auto maxLuminance = _dataset->stars[0].luminance;
    
    for (auto& star : _dataset->stars)
    {
        star.luminance /= maxLuminance;
    }

    _dataset->starCount = _dataset->stars.size();

    auto valuableStarCount = 20;
    if (_dataset->stars.size() < valuableStarCount)
    {
        valuableStarCount = _dataset->stars.size();
    }
    else
    {
        const Star upperVal{ Rect {}, PointF{}, 0.5, 0 };
        auto brightCount = std::upper_bound(_dataset->stars.begin(), _dataset->stars.end(), upperVal, [](const Star& a, const Star& b) { return a.luminance > b.luminance; }) - std::begin(_dataset->stars);
        if (brightCount > valuableStarCount)
            valuableStarCount = brightCount;

        _dataset->stars.erase(std::begin(_dataset->stars) + valuableStarCount, std::end(_dataset->stars));
    }

    return _dataset;
}
