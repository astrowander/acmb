#ifndef GENERATINGRESULTHELPER_H
#define GENERATINGRESULTHELPER_H
#include "../Core/IParallel.h"
#include "./stacker.h"

template<PixelFormat pixelFormat>
class GeneratingResultHelper : public IParallel
{
    Stacker& _stacker;
    std::shared_ptr<Bitmap<pixelFormat>> _pBitmap;

    GeneratingResultHelper(Stacker& stacker)
    : IParallel(stacker._height)
    , _stacker(stacker)
    , _pBitmap(std::make_shared<Bitmap<pixelFormat>>(stacker._width, stacker._height))
    {

    }

    void Job(uint32_t i) override
    {
        using ChannelType = typename PixelFormatTraits<pixelFormat>::ChannelType;

        ChannelType* channels[ChannelCount(pixelFormat)] = {};
        StackedChannel* stackedChannels[ChannelCount(pixelFormat)] = {};

        for (uint32_t ch = 0; ch < ChannelCount(pixelFormat); ++ch)
        {
            channels[ch] = &_pBitmap->GetScanline(i)[ch];
            stackedChannels[ch] = &_stacker._stacked[i * _stacker._width * ChannelCount(pixelFormat) + ch];
        }

        for (uint32_t x = 0; x < _stacker._width; ++x)        
        for (uint32_t ch = 0; ch < ChannelCount(pixelFormat); ++ch)
        {
            *channels[ch] = FastRound<ChannelType>(stackedChannels[ch]->mean);
            channels[ch] += ChannelCount(pixelFormat);
            stackedChannels[ch] += ChannelCount(pixelFormat);
        }            
    }

public:
    static std::shared_ptr<Bitmap<pixelFormat>> GenerateResult(Stacker& stacker)
    {
        GeneratingResultHelper helper(stacker);
        helper.DoParallelJobs();
        return helper._pBitmap;
    }
};

#endif