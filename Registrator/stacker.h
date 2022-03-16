#ifndef STACKER_H
#define STACKER_H
#include <memory>
#include <vector>
#include <string>

#include "../Core/bitmap.h"
#include "../Geometry/triangle.h"
#include "../AGG/agg_trans_affine.h"
#include "../Tests/test.h"
#include "registrator.h"
#include <array>

class ImageDecoder;
struct AlignmentDataset;
struct DatasetTiles;

struct StackedChannel
{
    float mean = 0;
    float dev = 0;
    uint16_t n = 1;
};

struct StackingDatum
{
    std::shared_ptr<ImageDecoder> pDecoder;
    std::vector<std::vector<Star>> stars;
    uint32_t totalStarCount;
};

class Stacker
{
    using TriangleTransformPair = std::pair<Triangle, agg::trans_affine>;
    using GridCell = std::vector<TriangleTransformPair>;
    using Grid = std::vector<GridCell>;

    Grid _grid;
    std::vector<StackingDatum> _stackingData;
    std::vector<StackedChannel> _stacked;
    uint32_t _hTileCount = 0;
    uint32_t _vTileCount = 0;

    uint32_t _width = 0;
    uint32_t _height = 0;

    static const size_t gridSize = 100;
    uint32_t _gridWidth = 0;
    uint32_t _gridHeight = 0;

    double _alignmentError = 2.0;

    bool _enableDeaberration;

    template<PixelFormat pixelFormat>
    void AddFirstBitmapToStack(std::shared_ptr<Bitmap<pixelFormat>> pBitmap)
    {
        auto stackedChannel = &_stacked[0];
        auto pSourceChannel = pBitmap->GetScanline(0);

        for (uint32_t i = 0; i < _height * _width * ChannelCount(pixelFormat); ++i)
        {
            (*stackedChannel++).mean = *pSourceChannel++;
        }
    }

    void ChooseTriangle(PointF p, std::pair<Triangle, agg::trans_affine>& lastPair, const GridCell& trianglePairs)
    {
        if (lastPair.first.IsPointInside(p))
            return;
        
        for (const auto& pair : trianglePairs)
        {
            if (pair.first.IsPointInside(p))
            {
                lastPair = pair;
                return;
            }
        }

        lastPair.second = agg::trans_affine_null();
    }

    template<PixelFormat pixelFormat>
    void AddBitmapToStackWithoutAlignment(std::shared_ptr<Bitmap<pixelFormat>> pBitmap)
    {
        using ChannelType = typename PixelFormatTraits<pixelFormat>::ChannelType;
        auto stackedChannel = &_stacked[0];
        auto pSourceChannel = pBitmap->GetScanline(0);

        for (uint32_t i = 0; i < _height * _width * ChannelCount(pixelFormat); ++i)
        {
            auto& mean = stackedChannel->mean;
            auto& dev = stackedChannel->dev;
            auto& n = stackedChannel->n;
            auto sigma = sqrt(dev);
            const auto kappa = 3.0;
            ChannelType sourceChannel = *pSourceChannel;

            if (n <= 5 || fabs(mean - sourceChannel) < kappa * sigma)
            {
                dev = n * (dev + (sourceChannel - mean) * (sourceChannel - mean) / (n + 1)) / (n + 1);
                mean = FitToBounds((n * mean + sourceChannel) / (n + 1), 0.0f, static_cast<float>(std::numeric_limits<typename PixelFormatTraits<pixelFormat>::ChannelType>::max()));
                ++n;
            }

            ++stackedChannel;
        }
    }

    template<PixelFormat pixelFormat>
    void AddBitmapToStack(std::shared_ptr<Bitmap<pixelFormat>> pBitmap)
    {
        StackedChannel* stackedChannels[ChannelCount(pixelFormat)] = {};
        TriangleTransformPair lastPair;

        for (uint32_t y = 0; y < _height; ++y)
        {
            for (uint32_t ch = 0; ch < ChannelCount(pixelFormat); ++ch)
            {
                stackedChannels[ch] = &_stacked[y * _width * ChannelCount(pixelFormat) + ch];
            }

            for (uint32_t x = 0; x < _width; ++x)
            {
                PointF p { static_cast<double>(x), static_cast<double>(y) };

                size_t hGridIndex = x / gridSize;
                size_t vGridIndex = y / gridSize;
                
                if (!_grid.empty())
                {
                    ChooseTriangle(p, lastPair, _grid[vGridIndex * _gridWidth + hGridIndex]);
                    lastPair.second.transform(&p.x, &p.y);
                }

                for (uint32_t ch = 0; ch < ChannelCount(pixelFormat); ++ch)
                {
                    if ((_grid.empty() || lastPair.second != agg::trans_affine_null()) && p.x >= 0 && p.x <= _width - 1 && p.y >= 0 && p.y <= _height - 1)
                    {
                        auto interpolatedChannel = pBitmap->GetInterpolatedChannel(static_cast<float>(p.x), static_cast<float>(p.y), ch);
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

        ChannelType* channels[ChannelCount(pixelFormat)] = {};
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

    Stacker(std::vector<std::shared_ptr<ImageDecoder>> decoders, bool enableDeaberration = false);

    void Registrate(uint32_t hTileCount = 1, uint32_t vTileCount = 1, double threshold = 40, uint32_t minStarSize = 5, uint32_t maxStarSize = 25);
    std::shared_ptr<IBitmap> Stack(bool doAlignment);

    TEST_ACCESS(Stacker);

};

#endif // STACKER_H
