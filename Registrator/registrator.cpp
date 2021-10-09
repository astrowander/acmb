#include "registrator.h"

Registrator::Registrator(std::shared_ptr<ImageDecoder> pDecoder, double threshold, uint32_t minStarSize, uint32_t maxStarSize)
: _dataset(std::make_shared<AlignmentDataset>(pDecoder))
, _threshold(threshold)
, _minStarSize(minStarSize)
, _maxStarSize(maxStarSize)
{

}

std::shared_ptr<AlignmentDataset> Registrator::Registrate(std::shared_ptr<ImageDecoder> pDecoder, double threshold, uint32_t minStarSize, uint32_t maxStarSize)
{
    Registrator reg(pDecoder, threshold, minStarSize, maxStarSize);

    if (BytesPerChannel(pDecoder->GetPixelFormat()) == 1)
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

    return reg._dataset;
}
