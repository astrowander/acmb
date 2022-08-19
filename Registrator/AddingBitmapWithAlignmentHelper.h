#ifndef ADDINGBITMAPWITHALIGNMENTHELPER_H
#define ADDINGBITMAPWITHALIGNMENTHELPER_H
#include "../Core/IParallel.h"
#include "./stacker.h"

template<PixelFormat pixelFormat>
class AddingBitmapWithAlignmentHelper final: public IParallel
{
    static constexpr uint32_t channelCount = ChannelCount(pixelFormat);

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
        Stacker::TriangleTransformPair lastPair;
       
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

          
            if ((_stacker._grid.empty() || lastPair.second != agg::trans_affine_null()) && p.x >= 0 && p.x <= _stacker._width - 1 && p.y >= 0 && p.y <= _stacker._height - 1)
            {
                for (uint32_t ch = 0; ch < channelCount; ++ch)
                {
                    const auto interpolatedChannel = _pBitmap->GetInterpolatedChannel(static_cast<float>(p.x), static_cast<float>(p.y), ch);
                    const size_t index = i * _stacker._width * channelCount + x * channelCount + ch;
                    auto& mean = _stacker._means[index];
                    auto& dev = _stacker._devs[index];
                    auto& n = _stacker._counts[index];

                    auto sigma = sqrt(dev);
                    const auto kappa = 3.0;

                    if (n <= 5 || fabs(mean - interpolatedChannel) < kappa * sigma)
                    {
                        dev = n * (dev + (interpolatedChannel - mean) * (interpolatedChannel - mean) / (n + 1)) / (n + 1);

                        mean = std::clamp((n * mean + interpolatedChannel) / (n + 1), 0.0f, static_cast<float>(std::numeric_limits<typename PixelFormatTraits<pixelFormat>::ChannelType>::max()));
                        ++n;
                    }
                }
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
