#include "registrator.h"

Registrator::Registrator(std::shared_ptr<IBitmap> pBitmap, double threshold, uint32_t minStarSize, uint32_t maxStarSize)
: _pBitmap(pBitmap)
, _threshold(threshold)
, _minStarSize(minStarSize)
, _maxStarSize(maxStarSize)
{

}

std::vector<Star> Registrator::Registrate(std::shared_ptr<IBitmap> pBitmap, double threshold, uint32_t minStarSize, uint32_t maxStarSize)
{
    Registrator reg(pBitmap, threshold, minStarSize, maxStarSize);

    if (BytesPerChannel(pBitmap->GetPixelFormat()) == 1)
        reg.Registrate<PixelFormat::Gray8>();
    else
        reg.Registrate<PixelFormat::Gray16>();

    std::sort(reg._stars.begin(), reg._stars.end(), [](auto& a, auto& b) {return a.luminance > b.luminance;});
    auto maxLuminance = reg._stars[0].luminance;
    for (auto& star : reg._stars)
        star.luminance /= maxLuminance;

    return reg._stars;
}

void Rect::ExpandRight(uint32_t right)
{
    if (width < right - x + 1)
        width = right - x + 1;
}

void Rect::ExpandLeft(uint32_t left)
{
    if (left < x)
    {
        x = left;
        width += x - left;
    }
}

void Rect::ExpandDown(uint32_t bottom)
{
    if (bottom > y + height - 1)
    {
        height = bottom - y + 1;
    }
}

bool Rect::operator==(const Rect &rhs)
{
    return (x == rhs.x) && (y == rhs.y) && (width = rhs.width) && (height == rhs.height);
}

bool Rect::operator!=(const Rect &lhs)
{
    return !(*this == lhs);
}
