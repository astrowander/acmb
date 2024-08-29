#pragma once
#include "../Core/bitmap.h"
#include "../Geometry/rect.h"

ACMB_NAMESPACE_BEGIN

/// <summary>
/// Holds statisitcs of a channel
/// </summary>
struct HistogramStatistics
{
	//the dimmest pixel
	uint32_t min = std::numeric_limits<uint32_t>::max();
	//the biggest number pixels has this value
	uint32_t peak = 0;
	//the brightest pixel
	uint32_t max = 0;
	// the median value
	uint32_t median = 0;
	// one percent of pixels is brighter than this value
	uint32_t topPercentile = 0;

	// i% pixels has lower value than i-st element if this array
	//uint32_t centils[100] = {};
	// average value
	float mean = 0.0f;
	// standard deviation
	float dev = 0.0f;
};
/// <summary>
/// Count statistics of the image and builds the histogram
/// </summary>
class HistogramBuilder
{
public:
	using ChannelHistogram = std::vector<uint32_t>;

protected:
	IBitmapPtr _pBitmap;
	HistogramBuilder(IBitmapPtr pBitmap);

public:
	/// Creates instance with source bitmap and region of interest. If it is empty transform respects the whole bitmap	
	static std::shared_ptr<HistogramBuilder> Create( IBitmapPtr pBitmap);
	/// Builds histograms of the source image
	virtual void BuildHistogram( Rect roi = {} ) = 0;
	/// returns histogram of the given channel
	virtual const ChannelHistogram& GetChannelHistogram(uint32_t ch) const = 0;
	/// returns statistics of the given channel
	virtual const HistogramStatistics& GetChannelStatistics(uint32_t ch) const = 0;
};

ACMB_NAMESPACE_END
