#include "converter.h"

template<>
void ConvertPixel<PixelFormat::RGB24, PixelFormat::Gray8>(uint8_t* pSrcPixel, uint8_t* pDstPixel)
{
    auto r = *pSrcPixel++;
    auto g = *pSrcPixel++;
    auto b = *pSrcPixel;

    *pDstPixel = static_cast<uint8_t>(0.299 * r + 0.587 * g + 0.114 * b + 0.5);
}

template<>
void ConvertPixel<PixelFormat::RGB48, PixelFormat::Gray16>(uint16_t* pSrcPixel, uint16_t* pDstPixel)
{
    auto r = *pSrcPixel++;
    auto g = *pSrcPixel++;
    auto b = *pSrcPixel;

    *pDstPixel = static_cast<uint16_t>(0.299 * r + 0.587 * g + 0.114 * b + 0.5);
}

std::shared_ptr<IBitmap> Convert(std::shared_ptr<IBitmap> pSrcBitmap, PixelFormat pDstPixelFormat)
{
    if (!pSrcBitmap)
        throw std::invalid_argument("pSrcBitmap is null");

    if (pSrcBitmap->GetPixelFormat() == PixelFormat::RGB24 && pDstPixelFormat == PixelFormat::Gray8)
    {
        return ConvertBitmap<PixelFormat::RGB24, PixelFormat::Gray8>(std::static_pointer_cast<Bitmap<PixelFormat::RGB24>>(pSrcBitmap));
    }
    else if (pSrcBitmap->GetPixelFormat() == PixelFormat::RGB48 && pDstPixelFormat == PixelFormat::Gray16)
    {
        return ConvertBitmap<PixelFormat::RGB48, PixelFormat::Gray16>(std::static_pointer_cast<Bitmap<PixelFormat::RGB48>>(pSrcBitmap));
    }

    throw std::runtime_error("unsupported pixel format");
}
