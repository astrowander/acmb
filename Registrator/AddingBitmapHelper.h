#ifndef ADDINGBITMAPHELPER_H
#define ADDINGBITMAPHELPER_H
#include "./stacker.h"

template<PixelFormat pixelFormat>
class AddingBitmapHelper
{
    static constexpr uint32_t channelCount = ChannelCount(pixelFormat);

	Stacker& _stacker;
	std::shared_ptr<Bitmap<pixelFormat>> _pBitmap;

	AddingBitmapHelper(Stacker& stacker, std::shared_ptr<Bitmap<pixelFormat>> pBitmap)
	: _stacker(stacker)
	, _pBitmap(pBitmap)
	{

	}

	void Job(uint32_t i)
	{
        using ChannelType = typename PixelFormatTraits<pixelFormat>::ChannelType;

        for (uint32_t j = 0; j < _stacker._width * channelCount; ++j)
        {
            const auto index = i * _stacker._width * channelCount + j;
            auto& mean = _stacker._means[index];
            auto& dev = _stacker._devs[index];
            auto& n = _stacker._counts[index];
            auto& channel = _pBitmap->GetScanline(0)[index];

            const auto sigma = sqrt(dev);
            const auto kappa = 3.0;                    

            if (n <= 5 || fabs(mean - channel) < kappa * sigma)
            {
                dev = n * (dev + (channel - mean) * (channel - mean) / (n + 1)) / (n + 1);
                mean = std::clamp((n * mean + channel) / (n + 1), 0.0f, static_cast<float>(std::numeric_limits<ChannelType>::max()));
                ++n;
            }
        }
	}

public:

    static void Run(Stacker& stacker, std::shared_ptr<Bitmap<pixelFormat>> pBitmap)
    {
        AddingBitmapHelper helper(stacker, pBitmap);
        oneapi::tbb::parallel_for( oneapi::tbb::blocked_range<int>( 0, pBitmap->GetHeight() ), [&helper] ( const oneapi::tbb::blocked_range<int>& range )
        {
            for ( int i = range.begin(); i < range.end(); ++i )
            {
                helper.Job( i );
            }
        } );
    }
};

#endif
