#include "ChannelEqualizer.h"

BaseChannelEqualizer::BaseChannelEqualizer(IBitmapPtr pSrcBitmap)
: BaseTransform(pSrcBitmap)
, IParallel(pSrcBitmap->GetHeight())
{}

std::shared_ptr<BaseChannelEqualizer> BaseChannelEqualizer::Create(IBitmapPtr pSrcBitmap, const std::vector<double>& multipliers)
{
	if (multipliers.size() != ChannelCount(pSrcBitmap->GetPixelFormat()))
		throw std::invalid_argument("Multiplier count must be equal to channel count");

	switch (pSrcBitmap->GetPixelFormat())
	{
	case PixelFormat::Gray8:
		return std::make_shared<ChannelEqualizer<PixelFormat::Gray8>>(pSrcBitmap, std::array<double, 1> { multipliers[0] });
	case PixelFormat::Gray16:
		return std::make_shared<ChannelEqualizer<PixelFormat::Gray16>>(pSrcBitmap, std::array<double, 1> { multipliers[0] });
	case PixelFormat::RGB24:
		return std::make_shared<ChannelEqualizer<PixelFormat::RGB24>>(pSrcBitmap, std::array<double, 3> { multipliers[0], multipliers[1], multipliers[2] });
	case PixelFormat::RGB48:
		return std::make_shared<ChannelEqualizer<PixelFormat::RGB48>>(pSrcBitmap, std::array<double, 3> { multipliers[0], multipliers[1], multipliers[2] });
	default:
		throw std::runtime_error("pixel format should be known");
	}
}
