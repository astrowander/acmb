#pragma once
#include "basetransform.h"

ACMB_NAMESPACE_BEGIN
/// <summary>
/// Divides the source image by the other one
/// </summary>
class BitmapDivisor : public BaseTransform
{
public:
    using Settings = IBitmapPtr;
protected:
    IBitmapPtr _pDivisor;
    BitmapDivisor( IBitmapPtr pSrcBitmap, IBitmapPtr pDivisor );

public:
    /// Creates an instance with source bitmap and bitmap that will be a divisor
    static std::shared_ptr<BitmapDivisor> Create( IBitmapPtr pSrcBitmap, IBitmapPtr pDivisor );
    /// Creates an instance with pixel format of source bitmap (need to set it later) and bitmap that will a divisor
    static std::shared_ptr<BitmapDivisor> Create( PixelFormat srcPixelFormat, IBitmapPtr pDivisor );
    /// Divides the source image by the other one
    static IBitmapPtr Divide( IBitmapPtr pSrcBitmap, IBitmapPtr pDivisor );
};

ACMB_NAMESPACE_END
