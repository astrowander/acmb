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
	// i% pixels has lower value than i-st element if this array
	uint32_t centils[100] = {};
	// average value
	float mean;
	// standard deviation
	float dev;
};
/// <summary>
/// Count statistics of the image and builds the histogram
/// </summary>
class HistorgamBuilder
{
public:
	using ChannelHistogram = std::vector<uint32_t>;
	using Settings = Rect;

protected:
	IBitmapPtr _pBitmap;
	Rect _roi;
	HistorgamBuilder(IBitmapPtr pBitmap, const Rect& roi);

public:
	/// Creates instance with source bitmap and region of interest. If it is empty transform respects the whole bitmap	
	static std::shared_ptr<HistorgamBuilder> Create( IBitmapPtr pBitmap, const Rect& roi = {});
	/// Creates instance with source pixel format and region of interest. If it is empty transform respects the whole bitmap
	/// Source bitmap must be set later
	static std::shared_ptr<HistorgamBuilder> Create( PixelFormat pixelFormat, const Rect& roi = {} );
	/// Builds histograms of the source image
	virtual void BuildHistogram() = 0;
	/// returns histogram of the given channel
	virtual const ChannelHistogram& GetChannelHistogram(uint32_t ch) const = 0;
	/// returns statistics of the given channel
	virtual const HistogramStatistics& GetChannelStatistics(uint32_t ch) const = 0;
};

ACMB_NAMESPACE_END
