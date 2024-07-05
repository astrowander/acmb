#pragma once

#include "basetransform.h"

ACMB_NAMESPACE_BEGIN

class MedianBlurTransform : public BaseTransform
{
public:
    using Settings = int;
protected:
    int _kernelSize{ 0 };
    MedianBlurTransform( IBitmapPtr pSrcBitmap, int kernelSize );

public:
    static std::shared_ptr<MedianBlurTransform> Create( IBitmapPtr pSrcBitmap, int kernelSize );
    static std::shared_ptr<MedianBlurTransform> Create( PixelFormat pixelFormat, int kernelSize );
    static IBitmapPtr MedianBlur( IBitmapPtr pSrcBitmap, int kernelSize );
};

ACMB_NAMESPACE_END
