#pragma once
#include "../Core/bitmap.h"
#include "../Core/IParallel.h"
#include <array>

struct HistogramStatistics
{
	uint32_t min = std::numeric_limits<uint32_t>::max();
	uint32_t peak = 0;
	uint32_t max = 0;
};

class BaseHistorgamBuilder : public IParallel
{
public:
	using ChannelHistogram = std::vector<uint32_t>;

protected:
	IBitmapPtr _pBitmap;
	BaseHistorgamBuilder(IBitmapPtr pBitmap);

public:
	static std::shared_ptr<BaseHistorgamBuilder> Create(IBitmapPtr pBitmap);

	virtual void BuildHistogram() = 0;
	virtual const ChannelHistogram& GetChannelHistogram(uint32_t ch) const = 0;
	virtual const HistogramStatistics& GetChannelStatistics(uint32_t ch) const = 0;
};



template <PixelFormat pixelFormat>
class HistogramBuilder : public BaseHistorgamBuilder
{
	using ChannelType = typename PixelFormatTraits<pixelFormat>::ChannelType;
	static const auto channelCount = PixelFormatTraits<pixelFormat>::channelCount;
	static const uint32_t channelMax = PixelFormatTraits<pixelFormat>::channelMax;	

	std::array<ChannelHistogram, channelCount> _histograms;
	std::array<HistogramStatistics, channelCount> _statistics;

	void Job(uint32_t i) override
	{
		std::scoped_lock lock( _mutex );
		for (uint32_t ch = 0; ch < channelCount; ++ch)
		{
			auto pBitmap = std::static_pointer_cast<Bitmap<pixelFormat>>(_pBitmap);
			auto pChannel = pBitmap->GetScanline(i) + ch;

			for (uint32_t x = 0; x < pBitmap->GetWidth(); ++x)
			{
				ChannelType val = *pChannel;
				++_histograms[ch][val];
				if (val < _statistics[ch].min)
				{
					_statistics[ch].min = val;
				}
				if (val > _statistics[ch].max)
				{
					_statistics[ch].max = val;
				}
				if   ( _histograms[ch][val] >  _histograms[ch][_statistics[ch].peak] ||
					 ( _histograms[ch][val] == _histograms[ch][_statistics[ch].peak] && val > _statistics[ch].peak ) )
				{
					_statistics[ch].peak = val;
				}
				pChannel += channelCount;
			}
		}
	}

public:
	HistogramBuilder(IBitmapPtr pBitmap)
	: BaseHistorgamBuilder(pBitmap)
	{}

	void BuildHistogram() override
	{
		for (uint32_t ch = 0; ch < channelCount; ++ch)
			_histograms[ch].resize(channelMax + 1);

		DoParallelJobs();
	}

	const ChannelHistogram& GetChannelHistogram(uint32_t ch) const override
	{
		return _histograms[ch];
	}

	const HistogramStatistics& GetChannelStatistics(uint32_t ch) const
	{
		return _statistics[ch];
	}

};
