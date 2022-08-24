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

BaseConverter::BaseConverter(IBitmapPtr pSrcBitmap)
: BaseTransform(pSrcBitmap)
{

}

std::shared_ptr<BaseConverter> BaseConverter::Create(IBitmapPtr pSrcBitmap, PixelFormat dstPixelFormat)
{
    if (!pSrcBitmap)
        throw std::invalid_argument("pSrcBitmap is null");

    if (pSrcBitmap->GetPixelFormat() == PixelFormat::RGB24 && dstPixelFormat == PixelFormat::Gray8)
    {
        return std::make_shared<Converter<PixelFormat::RGB24, PixelFormat::Gray8>>(pSrcBitmap);
    }
    else if (pSrcBitmap->GetPixelFormat() == PixelFormat::RGB48 && dstPixelFormat == PixelFormat::Gray16)
    {
        return std::make_shared<Converter<PixelFormat::RGB48, PixelFormat::Gray16>>(pSrcBitmap);
    }

    throw std::runtime_error("unsupported pixel format");
}

IBitmapPtr BaseConverter::Convert(IBitmapPtr pSrcBitmap, PixelFormat dstPixelFormat)
{
    return Create(pSrcBitmap, dstPixelFormat)->RunAndGetBitmap();
}
