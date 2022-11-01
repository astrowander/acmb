#pragma once
#include "basetransform.h"
#include "../Geometry/size.h"
ACMB_NAMESPACE_BEGIN
/// <summary>
/// Resizes image to the arbitrary size
/// </summary>
class ResizeTransform : public BaseTransform
{
public:
    using Settings = Size;

protected:
    Size _dstSize;
    ResizeTransform( std::shared_ptr<IBitmap> pSrcBitmap, Size dstSize );
public:
    /// Creates an instance with given bitmap and bin size
    static std::shared_ptr<ResizeTransform> Create( IBitmapPtr pSrcBitmap, Size dstSize );
    /// Creates an instance with given pixel format (need to set source bitmap later) and bin size
    static std::shared_ptr<ResizeTransform> Create( PixelFormat pixelFormat, Size dstSize );
    /// Resizes image to the arbitrary size
    static IBitmapPtr Resize( IBitmapPtr pSrcBitmap, Size dstSize );

    /// returns size of destination image
    virtual void CalcParams( std::shared_ptr<ImageParams> pParams ) override;
};

ACMB_NAMESPACE_END
