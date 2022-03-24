#include "binningtransform.h"

IBinningTransform::IBinningTransform(std::shared_ptr<IBitmap> pSrcBitmap, Size bin)
: BaseTransform(pSrcBitmap)
, _bin(bin)
{
    if (_bin.width == 0 || _bin.height == 0)
        throw std::invalid_argument("zero bin size");
}

std::shared_ptr<IBinningTransform> IBinningTransform::Create(std::shared_ptr<IBitmap> pSrcBitmap, Size bin)
{
    if (!pSrcBitmap)
        throw std::invalid_argument("pSrcBitmap is null");

    switch (pSrcBitmap->GetPixelFormat())
    {
    case PixelFormat::Gray8:
        return std::make_shared<BinningTransform<PixelFormat::Gray8>>(pSrcBitmap, bin);
    case PixelFormat::Gray16:
        return std::make_shared<BinningTransform<PixelFormat::Gray16>>(pSrcBitmap, bin);
    case PixelFormat::RGB24:
        return std::make_shared<BinningTransform<PixelFormat::RGB24>>(pSrcBitmap, bin);
    case PixelFormat::RGB48:
        return std::make_shared<BinningTransform<PixelFormat::RGB48>>(pSrcBitmap, bin);
    default:
        throw std::runtime_error("Pixel format must be known");
    }
}
