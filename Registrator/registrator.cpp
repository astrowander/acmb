#include "registrator.h"

Registrator::Registrator(std::shared_ptr<IBitmap> pBitmap, double threshold, uint32_t minStarSize, uint32_t maxStarSize)
: _dataset(std::make_shared<AlignmentDataset>())
,  _pBitmap(pBitmap)
, _threshold(threshold)
, _minStarSize(minStarSize)
, _maxStarSize(maxStarSize)
{

}

std::shared_ptr<AlignmentDataset> Registrator::Registrate(std::shared_ptr<IBitmap> pBitmap, double threshold, uint32_t minStarSize, uint32_t maxStarSize)
{
    Registrator reg(pBitmap, threshold, minStarSize, maxStarSize);

    if (BytesPerChannel(pBitmap->GetPixelFormat()) == 1)
        reg.Registrate<PixelFormat::Gray8>();
    else
        reg.Registrate<PixelFormat::Gray16>();

    std::sort(reg._dataset->stars.begin(), reg._dataset->stars.end(), [](auto& a, auto& b) {return a.luminance > b.luminance;});
    auto maxLuminance = reg._dataset->stars[0].luminance;
    //Point p0 = reg._dataset->stars[0].rect.GetOrigin();
    //reg._dataset->transform.translate(p0.x, p0.y);

    for (auto& star : reg._dataset->stars)
    {
        star.luminance /= maxLuminance;
        //star.rect.Translate(-p0.x, -p0.y);
    }

    if (reg._dataset->stars.size() < reg._dataset->valuableStarCount)
    {
        reg._dataset->valuableStarCount = reg._dataset->stars.size();
    }
    else
    {
        const Star upperVal {Rect {}, PointF{}, 0.5, 0};
        auto brightCount = std::upper_bound(reg._dataset->stars.begin(), reg._dataset->stars.end(), upperVal, [](const Star& a, const Star& b) { return a.luminance > b.luminance; }) - std::begin(reg._dataset->stars);
        if (brightCount > reg._dataset->valuableStarCount)
            reg._dataset->valuableStarCount = brightCount;
    }

    return reg._dataset;
}
