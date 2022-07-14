#ifndef ADDINGBITMAPHELPER_H
#define ADDINGBITMAPHELPER_H
#include "../Core/IParallel.h"
#include "./stacker.h"

template<PixelFormat pixelFormat>
class AddingBitmapHelper : public IParallel
{
	Stacker& _stacker;
	std::shared_ptr<Bitmap<pixelFormat>> _pBitmap;

	AddingBitmapHelper(Stacker& stacker, std::shared_ptr<Bitmap<pixelFormat>> pBitmap)
	: IParallel(pBitmap->GetHeight())
    , _stacker(stacker)
	, _pBitmap(pBitmap)
	{

	}

	void Job(uint32_t i) override
	{
        using ChannelType = typename PixelFormatTraits<pixelFormat>::ChannelType;
        auto pMean = &_stacker._means[i * _stacker._width * ChannelCount(pixelFormat)];
        auto pDev = &_stacker._devs[i * _stacker._width * ChannelCount(pixelFormat)];
        auto pCount = &_stacker._counts[i * _stacker._width * ChannelCount(pixelFormat)];
        auto pSourceChannel = &_pBitmap->GetScanline(i)[0];

        for (uint32_t j = 0; j < _stacker._width * ChannelCount(pixelFormat); ++j)
        {
            auto sigma = sqrt(*pDev);
            const auto kappa = 3.0;

            if (*pCount <= 5 || fabs(*pMean - *pSourceChannel) < kappa * sigma)
            {
                *pDev = *pCount * (*pDev + (*pSourceChannel - *pMean) * (*pSourceChannel - *pMean) / (*pCount + 1)) / (*pCount + 1);
                *pMean = FitToBounds((*pCount * *pMean + *pSourceChannel) / (*pCount + 1), 0.0f, static_cast<float>(std::numeric_limits<typename PixelFormatTraits<pixelFormat>::ChannelType>::max()));
                ++(*pCount);
            }

            ++pSourceChannel;
            ++pMean;
            ++pDev;
            ++pCount;
        }
	}

public:

    static void Run(Stacker& stacker, std::shared_ptr<Bitmap<pixelFormat>> pBitmap)
    {
        AddingBitmapHelper helper(stacker, pBitmap);
        helper.DoParallelJobs();
    }
};

#endif
