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

    return reg._stars;
}

/*template <PixelFormat pixelFormat>
void Registrator::Registrate()
{
    auto pGrayBitmap = Convert(_pBitmap, BytesPerChannel(_pBitmap->GetPixelFormat()) == 1 ? PixelFormat::Gray8 : PixelFormat::Gray16);

}*/

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
    if (height < bottom - y - 1)
    {
        height = bottom - y - 1;
    }
}
