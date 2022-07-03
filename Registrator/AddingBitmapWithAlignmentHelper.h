#ifndef ADDINGBITMAPWITHALIGNMENTHELPER_H
#define ADDINGBITMAPWITHALIGNMENTHELPER_H
#include "../Core/IParallel.h"
#include "./stacker.h"

template<PixelFormat pixelFormat>
class AddingBitmapWithAlignmentHelper : public IParallel
{
    Stacker& _stacker;
    std::shared_ptr<Bitmap<pixelFormat>> _pBitmap;

    AddingBitmapWithAlignmentHelper(Stacker& stacker, std::shared_ptr<Bitmap<pixelFormat>> pBitmap)
        : IParallel(pBitmap->GetHeight())
        , _stacker(stacker)
        , _pBitmap(pBitmap)
    {

    }

    void Job(uint32_t i) override
    {
        StackedChannel* stackedChannels[ChannelCount(pixelFormat)] = {};
        Stacker::TriangleTransformPair lastPair;

       
        for (uint32_t ch = 0; ch < ChannelCount(pixelFormat); ++ch)
        {
            stackedChannels[ch] = &_stacker._stacked[i * _stacker._width * ChannelCount(pixelFormat) + ch];
        }

        for (uint32_t x = 0; x < _stacker._width; ++x)
        {
            PointF p{ static_cast<double>(x), static_cast<double>(i) };

            size_t hGridIndex = x / _stacker.gridSize;
            size_t vGridIndex = i / _stacker.gridSize;

            if (!_stacker._grid.empty())
            {
                _stacker.ChooseTriangle(p, lastPair, _stacker._grid[vGridIndex * _stacker._gridWidth + hGridIndex]);
                lastPair.second.transform(&p.x, &p.y);
            }

            for (uint32_t ch = 0; ch < ChannelCount(pixelFormat); ++ch)
            {
                if ((_stacker._grid.empty() || lastPair.second != agg::trans_affine_null()) && p.x >= 0 && p.x <= _stacker._width - 1 && p.y >= 0 && p.y <= _stacker._height - 1)
                {
                    auto interpolatedChannel = _pBitmap->GetInterpolatedChannel(static_cast<float>(p.x), static_cast<float>(p.y), ch);
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

public:

    static void Run(Stacker& stacker, std::shared_ptr<Bitmap<pixelFormat>> pBitmap)
    {
        AddingBitmapWithAlignmentHelper helper(stacker, pBitmap);
        helper.DoParallelJobs();
    }
};

#endif
