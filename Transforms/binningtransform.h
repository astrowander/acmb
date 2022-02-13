#ifndef BINNINGTRANSFORM_H
#define BINNINGTRANSFORM_H
#include "basetransform.h"
#include <array>
#include <algorithm>
#include <stdexcept>

template<uint32_t hBinSize, uint32_t vBinSize>
class IBinningTransform : public BaseTransform
{
public:
    IBinningTransform(std::shared_ptr<IBitmap> pSrcBitmap)
    : BaseTransform(pSrcBitmap)
    {
    }

    static std::shared_ptr<IBinningTransform<hBinSize, vBinSize>> Create(std::shared_ptr<IBitmap> pSrcBitmap);
};

template<PixelFormat pixelFormat, uint32_t hBinSize, uint32_t vBinSize>
class BinningTransform : public IBinningTransform<hBinSize, vBinSize>
{
    using ChannelType = typename PixelFormatTraits<pixelFormat>::ChannelType;
    static constexpr auto channelCount = PixelFormatTraits<pixelFormat>::channelCount;

    std::array<ChannelType, hBinSize* vBinSize> _buf = {};

public:
    BinningTransform(std::shared_ptr<IBitmap> pSrcBitmap)
    : IBinningTransform<hBinSize, vBinSize>(pSrcBitmap)
    {
    }

    void Run() override
    {
        auto pSrcBitmap = std::static_pointer_cast<Bitmap<pixelFormat>>(this->_pSrcBitmap);
        auto w = pSrcBitmap->GetWidth() / hBinSize;
        auto h = pSrcBitmap->GetHeight() / vBinSize;
        this->_pDstBitmap.reset(new Bitmap<pixelFormat>(w , h));
        auto pDstBitmap = std::static_pointer_cast<Bitmap<pixelFormat>>(this->_pDstBitmap);

        for (uint32_t i = 0; i < h; ++i)
        {
            auto pSrcPixel = pSrcBitmap->GetScanline(i * vBinSize);
            auto pDstPixel = pDstBitmap->GetScanline(i);

            for (uint32_t j = 0; j < w; ++j)
            {
                ProcessPixel(pSrcPixel, pDstPixel);
                pSrcPixel += channelCount * hBinSize;
                pDstPixel += channelCount;
            }
        }
    }

private:
    void ProcessPixel(ChannelType* pSrcPixel, ChannelType* pDstPixel)
    {
        for (uint32_t ch = 0; ch < channelCount; ++ch)
        {
            for (uint32_t i = 0; i < vBinSize; ++i)
            for (uint32_t j = 0; j < hBinSize; ++j)
            {
                _buf[i * hBinSize + j] = pSrcPixel[(this->_pSrcBitmap->GetWidth() * i + j) * channelCount + ch];
            }

            auto median = _buf.begin() + _buf.size() / 2;
            std::nth_element(_buf.begin(), median, _buf.end());
            pDstPixel[ch] = *median;
        }
    }

};

template<uint32_t hBinSize, uint32_t vBinSize>
std::shared_ptr<IBinningTransform<hBinSize, vBinSize>> IBinningTransform<hBinSize, vBinSize>::Create(std::shared_ptr<IBitmap> pSrcBitmap)
{
    if (!pSrcBitmap)
        throw std::invalid_argument("pSrcBitmap is null");

    switch (pSrcBitmap->GetPixelFormat())
    {
        case PixelFormat::Gray8:
            return std::make_shared<BinningTransform<PixelFormat::Gray8, hBinSize, vBinSize>>(pSrcBitmap);
        case PixelFormat::Gray16:
            return std::make_shared<BinningTransform<PixelFormat::Gray16, hBinSize, vBinSize>>(pSrcBitmap);
        case PixelFormat::RGB24:
            return std::make_shared<BinningTransform<PixelFormat::RGB24, hBinSize, vBinSize>>(pSrcBitmap);
        case PixelFormat::RGB48:
            return std::make_shared<BinningTransform<PixelFormat::RGB48, hBinSize, vBinSize>>(pSrcBitmap);
        default:
            throw std::runtime_error("Pixel format must be known");
    }
}



#endif // BINNINGTRANSFORM_H
