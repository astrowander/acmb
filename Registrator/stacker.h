#ifndef STACKER_H
#define STACKER_H
#include <memory>
#include <vector>
#include <string>

#include "Core/bitmap.h"
#include "Geometry/rect.h"
#include "AGG/agg_trans_affine.h"

class ImageDecoder;
struct AlignmentDataset;

struct StackedChannel
{
    float mean = 0;
    float dev = 0;
};

class Stacker
{
    std::vector<std::shared_ptr<ImageDecoder>> _decoders;
    std::vector<std::shared_ptr<AlignmentDataset>> _datasets;
    std::vector<StackedChannel> _stacked;

    uint32_t _width;
    uint32_t _height;

    template<PixelFormat pixelFormat>
    void AddBitmapToStack(std::shared_ptr<Bitmap<pixelFormat>> pBitmap, uint32_t n, const agg::trans_affine& transform)
    {
        StackedChannel* stackedChannels[ChannelCount(pixelFormat)];

        for (uint32_t ch = 0; ch < ChannelCount(pixelFormat); ++ch)
        {
            stackedChannels[ch] = &_stacked[ch];
        }

        for (uint32_t y = 0; y < _height; ++y)
        for (uint32_t x = 0; x < _width; ++x)
        {
            PointF targetPoint {static_cast<float>(x), static_cast<float>(y)};
            transform.transform(&targetPoint.x, &targetPoint.y);

            for (uint32_t ch = 0; ch < ChannelCount(pixelFormat); ++ch)
            {
                if (targetPoint.x >= 0 && targetPoint.x <= _width - 1 && targetPoint.y >= 0 && targetPoint.y <= _height - 1)
                {
                    auto interpolatedChannel = pBitmap->GetInterpolatedChannel(targetPoint.x , targetPoint.y, ch);
                    auto& mean = stackedChannels[ch]->mean;
                    auto& dev = stackedChannels[ch]->dev;

                    dev = n * (dev + (interpolatedChannel - mean) * (interpolatedChannel - mean) / (n + 1)) / (n + 1);
                    mean = (n * mean + interpolatedChannel) / (n + 1);
                }

                stackedChannels[ch] += ChannelCount(pixelFormat);
            }
        }
    }

    template<PixelFormat pixelFormat>
    std::shared_ptr<Bitmap<pixelFormat>> GetStackedBitmap()
    {
        using ChannelType = typename PixelFormatTraits<pixelFormat>::ChannelType;
        auto pRes = std::make_shared<Bitmap<pixelFormat>>(_width, _height);

        if (_width == 0 || _height == 0)
            return pRes;

        ChannelType* channels[ChannelCount(pixelFormat)];
        StackedChannel* stackedChannels[ChannelCount(pixelFormat)];

        for (uint32_t ch = 0; ch < ChannelCount(pixelFormat); ++ch)
        {
            channels[ch] = &pRes->GetScanline(0)[ch];
            stackedChannels[ch] = &_stacked[ch];
        }

        for (uint32_t y = 0; y < _height; ++y)
        for (uint32_t x = 0; x < _width; ++x)
        {
            for (uint32_t ch = 0; ch < ChannelCount(pixelFormat); ++ch)
            {
                *channels[ch] = FastRound<ChannelType>(stackedChannels[ch]->mean);
                channels[ch] += ChannelCount(pixelFormat);
                stackedChannels[ch] += ChannelCount(pixelFormat);
            }
        }

        return pRes;
    }

public:
    Stacker(std::vector<std::shared_ptr<ImageDecoder>> decoders);
    std::shared_ptr<IBitmap> Stack();

};

#endif // STACKER_H
