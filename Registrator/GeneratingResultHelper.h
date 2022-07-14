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

        ChannelType* pChannel = &_pBitmap->GetScanline(i)[0];
        float* pMean = &_stacker._means[i * _stacker._width * ChannelCount(pixelFormat)];

        for (uint32_t x = 0; x < _stacker._width; ++x)        
        for (uint32_t ch = 0; ch < ChannelCount(pixelFormat); ++ch)
        {
            *pChannel = FastRound<ChannelType>(*pMean);
            ++pMean;
            ++pChannel;
        }            
    }

public:
    static void Run(Stacker& stacker, std::shared_ptr<Bitmap<pixelFormat>> pBitmap)
    {
        GeneratingResultHelper helper(stacker);
        helper.DoParallelJobs();
        pBitmap->SetData(helper._pBitmap->GetData());
    }
};

#endif