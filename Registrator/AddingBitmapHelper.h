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
        auto stackedChannel = &_stacker._stacked[i * _stacker._width * ChannelCount(pixelFormat)];
        auto pSourceChannel = _pBitmap->GetScanline(i);

        for (uint32_t j = 0; j < _stacker._width * ChannelCount(pixelFormat); ++j)
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
            ++pSourceChannel;
        }
	}

public:

    static void AddBitmap(Stacker& stacker, std::shared_ptr<Bitmap<pixelFormat>> pBitmap)
    {
        AddingBitmapHelper helper(stacker, pBitmap);
        helper.DoParallelJobs();
    }
};

#endif
