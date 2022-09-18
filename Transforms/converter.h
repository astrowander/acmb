#pragma once
#include "../Core/bitmap.h"
#include <stdexcept>
#include "basetransform.h"
ACMB_NAMESPACE_BEGIN

class Converter : public BaseTransform
{
protected:
    Converter(IBitmapPtr pSrcBitmap);

public:
    static std::shared_ptr<Converter> Create(IBitmapPtr pSrcBitmap, PixelFormat dstPixelFormat);
    static IBitmapPtr Convert(IBitmapPtr pSrcBitmap, PixelFormat dstPixelFormat);
};

ACMB_NAMESPACE_END


