#pragma once
#include "../Core/bitmap.h"
#include "../Geometry/rect.h"
#include <array>
#include <mutex>

struct HistogramStatistics
{
	uint32_t min = std::numeric_limits<uint32_t>::max();
	uint32_t peak = 0;
	uint32_t max = 0;
	uint32_t centils[100] = {};
	float mean;
	float dev;
};

class BaseHistorgamBuilder
{
public:
	using ChannelHistogram = std::vector<uint32_t>;

protected:
	IBitmapPtr _pBitmap;
	Rect _roi;
	BaseHistorgamBuilder(IBitmapPtr pBitmap, const Rect& roi);

public:
	static std::shared_ptr<BaseHistorgamBuilder> Create(IBitmapPtr pBitmap, const Rect& roi = {});

	virtual void BuildHistogram() = 0;
	virtual const ChannelHistogram& GetChannelHistogram(uint32_t ch) const = 0;
	virtual const HistogramStatistics& GetChannelStatistics(uint32_t ch) const = 0;
};



template <PixelFormat pixelFormat>
class HistogramBuilder final: public BaseHistorgamBuilder
{
	using ChannelType = typename PixelFormatTraits<pixelFormat>::ChannelType;
	static const auto channelCount = PixelFormatTraits<pixelFormat>::channelCount;
	static const uint32_t channelMax = PixelFormatTraits<pixelFormat>::channelMax;	

	std::array<ChannelHistogram, channelCount> _histograms;
	std::array<HistogramStatistics, channelCount> _statistics;

	std::mutex _mutex;

public:
	HistogramBuilder( IBitmapPtr pBitmap, const Rect& roi );

	void BuildHistogram() override;

	const ChannelHistogram& GetChannelHistogram( uint32_t ch ) const override;

	const HistogramStatistics& GetChannelStatistics( uint32_t ch ) const override;

};
