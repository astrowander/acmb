#ifndef STACKER_H
#define STACKER_H
#include <memory>
#include <vector>
#include <string>

#include "../Core/bitmap.h"
#include "../Geometry/rect.h"
#include "../AGG/agg_trans_affine.h"
#include "../Tests/test.h"
#include "registrator.h"

class ImageDecoder;
struct AlignmentDataset;
struct DatasetTiles;

struct StackedChannel
{
    float mean = 0;
    float dev = 0;
    uint16_t n = 0;
};

class Stacker
{
    std::vector<std::pair<std::shared_ptr<ImageDecoder>, std::shared_ptr<DatasetTiles>>> _decoderDatasetPairs;
    std::vector<StackedChannel> _stacked;

    uint32_t _width = 0;
    uint32_t _height = 0;
    uint32_t _hTiles = 0;
    uint32_t _vTiles = 1;

    template<PixelFormat pixelFormat>
    void AddBitmapToStack(std::shared_ptr<Bitmap<pixelFormat>> pBitmap, std::shared_ptr<DatasetTiles> pDatasetTiles)
    {
        StackedChannel* stackedChannels[ChannelCount(pixelFormat)];
        const auto tileWidth = pBitmap->GetWidth() / _hTiles;
        const auto tileHeight = pBitmap->GetHeight() / _vTiles;

        for (uint32_t y = 0; y < _height; ++y)
        {
            for (uint32_t ch = 0; ch < ChannelCount(pixelFormat); ++ch)
            {
                stackedChannels[ch] = &_stacked[y * _width * ChannelCount(pixelFormat) + ch];
            }

            const auto yTile = std::min(y / tileHeight, _vTiles - 1);

            for (uint32_t x = 0; x < _width; ++x)
            {
                const auto xTile = std::min(x / tileWidth, _hTiles - 1);
                const auto & transform = pDatasetTiles ? pDatasetTiles->datasets[yTile * _hTiles + xTile]->transform : agg::trans_affine();
                if (transform == agg::trans_affine_null())
                {
                    for (uint32_t ch = 0; ch < ChannelCount(pixelFormat); ++ch)
                        stackedChannels[ch] += ChannelCount(pixelFormat);
                    continue;
                }

                PointF targetPoint{ static_cast<double>(x), static_cast<double>(y) };
                transform.transform(&targetPoint.x, &targetPoint.y);

                for (uint32_t ch = 0; ch < ChannelCount(pixelFormat); ++ch)
                {
                    if (targetPoint.x >= 0 && targetPoint.x <= _width - 1 && targetPoint.y >= 0 && targetPoint.y <= _height - 1)
                    {
                        auto interpolatedChannel = pBitmap->GetInterpolatedChannel(static_cast<float>(targetPoint.x), static_cast<float>(targetPoint.y), ch);
                        auto& mean = stackedChannels[ch]->mean;
                        auto& dev = stackedChannels[ch]->dev;
                        auto& n = stackedChannels[ch]->n;
                        auto sigma = sqrt(dev);
                        const auto kappa = 3.0;

                        if (n <= 5 || fabs(mean - interpolatedChannel) < kappa * sigma)
                        {
                            dev = n * (dev + (interpolatedChannel - mean) * (interpolatedChannel - mean) / (n + 1)) / (n + 1);

                            mean = FitToBounds((n * mean + interpolatedChannel) / (n + 1), 0.0f, static_cast<float>(std::numeric_limits<typename PixelFormatTraits<pixelFormat>::ChannelType>::max()));
                            ++n;
                        }
                    }

                    stackedChannels[ch] += ChannelCount(pixelFormat);
                }
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

    void Registrate(double threshold = 40, uint32_t minStarSize = 5, uint32_t maxStarSize = 25, uint32_t hTiles = 1, uint32_t vTiles = 1);
    std::shared_ptr<IBitmap> Stack(bool doAlignment);

    TEST_ACCESS(Stacker);

};

#endif // STACKER_H
