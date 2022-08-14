#include "HistogramBuilder.h"

BaseHistorgamBuilder::BaseHistorgamBuilder(IBitmapPtr pBitmap, const Rect& roi)
: IParallel(roi.height ? roi.height : pBitmap->GetHeight())
, _pBitmap(pBitmap)
, _roi((roi.width&& roi.height) ? roi : Rect(0, 0, pBitmap->GetWidth(), pBitmap->GetHeight()))
{
	
}

std::shared_ptr<BaseHistorgamBuilder> BaseHistorgamBuilder::Create(IBitmapPtr pBitmap, const Rect& roi)
{
	switch (pBitmap->GetPixelFormat())
	{
	case PixelFormat::Gray8:
		return std::make_shared<HistogramBuilder<PixelFormat::Gray8>>(pBitmap, roi);
	case PixelFormat::Gray16:
		return std::make_shared<HistogramBuilder<PixelFormat::Gray16>>(pBitmap, roi);
	case PixelFormat::RGB24:
		return std::make_shared<HistogramBuilder<PixelFormat::RGB24>>(pBitmap, roi);
	case PixelFormat::RGB48:
		return std::make_shared<HistogramBuilder<PixelFormat::RGB48>>(pBitmap, roi);
	default:
		throw std::runtime_error("pixel format should be known");
	}
}