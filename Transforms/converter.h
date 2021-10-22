#ifndef CONVERTER_H
#define CONVERTER_H
#include "../Core/bitmap.h"
#include <stdexcept>

template<PixelFormat srcPixelFormat, PixelFormat dstPixelFormat>
void ConvertPixel(std::conditional_t<BytesPerChannel(srcPixelFormat) == 1, uint8_t*, uint16_t*>, std::conditional_t<BytesPerChannel(srcPixelFormat) == 1, uint8_t*, uint16_t*>)
{
    throw std::runtime_error("not implemented");
}

template<>
void ConvertPixel<PixelFormat::RGB24, PixelFormat::Gray8>(uint8_t* pSrcPixel, uint8_t* pDstPixel);

template<>
void ConvertPixel<PixelFormat::RGB48, PixelFormat::Gray8>(uint16_t* pSrcPixel, uint16_t* pDstPixel);

template<PixelFormat srcPixelFormat, PixelFormat dstPixelFormat>
auto ConvertBitmap(std::shared_ptr<Bitmap<srcPixelFormat>> pSrcBitmap)
{
    auto pDstBitmap = std::make_shared<Bitmap<dstPixelFormat>>(pSrcBitmap->GetWidth(), pSrcBitmap->GetHeight());
    for (uint32_t i = 0; i < pDstBitmap->GetHeight(); ++i)
    {
        auto pSrcScanline = pSrcBitmap->GetScanline(i);
        auto pDstScanline = pDstBitmap->GetScanline(i);

        for (uint32_t j = 0; j < pDstBitmap->GetWidth(); ++j)
        {
            ConvertPixel<srcPixelFormat, dstPixelFormat>(pSrcScanline, pDstScanline);
            pSrcScanline += ChannelCount(srcPixelFormat);
            pDstScanline += ChannelCount(dstPixelFormat);
        }
    }
    return pDstBitmap;
}

std::shared_ptr<IBitmap> Convert(std::shared_ptr<IBitmap> pSrcBitmap, PixelFormat pDstPixelFormat);

#endif // CONVERTER_H
