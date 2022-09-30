#pragma once
#include "../Core/bitmap.h"
#include "../Geometry/rect.h"

ACMB_NAMESPACE_BEGIN

struct HistogramStatistics
{
	uint32_t min = std::numeric_limits<uint32_t>::max();
	uint32_t peak = 0;
	uint32_t max = 0;
	uint32_t centils[100] = {};
	float mean;
	float dev;
};

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
	static std::shared_ptr<HistorgamBuilder> Create( IBitmapPtr pBitmap, const Rect& roi = {});
	static std::shared_ptr<HistorgamBuilder> Create( PixelFormat pixelFormat, const Rect& roi = {} );

	virtual void BuildHistogram() = 0;
	virtual const ChannelHistogram& GetChannelHistogram(uint32_t ch) const = 0;
	virtual const HistogramStatistics& GetChannelStatistics(uint32_t ch) const = 0;
};

ACMB_NAMESPACE_END
