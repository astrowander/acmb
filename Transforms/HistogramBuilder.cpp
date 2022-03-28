#include "HistogramBuilder.h"

BaseHistorgamBuilder::BaseHistorgamBuilder(IBitmapPtr pBitmap)
: IParallel(pBitmap->GetHeight())
, _pBitmap(pBitmap)
{}

std::shared_ptr<BaseHistorgamBuilder> BaseHistorgamBuilder::Create(IBitmapPtr pBitmap)
{
	switch (pBitmap->GetPixelFormat())
	{
	case PixelFormat::Gray8:
		return std::make_shared<HistogramBuilder<PixelFormat::Gray8>>(pBitmap);
	case PixelFormat::Gray16:
		return std::make_shared<HistogramBuilder<PixelFormat::Gray16>>(pBitmap);
	case PixelFormat::RGB24:
		return std::make_shared<HistogramBuilder<PixelFormat::RGB24>>(pBitmap);
	case PixelFormat::RGB48:
		return std::make_shared<HistogramBuilder<PixelFormat::RGB48>>(pBitmap);
	default:
		throw std::runtime_error("pixel format should be known");
	}
}