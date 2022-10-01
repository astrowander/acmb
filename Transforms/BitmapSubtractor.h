#pragma once
#include "basetransform.h"

ACMB_NAMESPACE_BEGIN
/// <summary>
/// Subtracts given image from the source
/// </summary>
class BitmapSubtractor : public BaseTransform
{
public:
    using Settings = IBitmapPtr;
protected:
    IBitmapPtr _pBitmapToSubtract;
    BitmapSubtractor( IBitmapPtr pSrcBitmap, IBitmapPtr pBitmapToSubtract );

public:
    /// Creates an instance with source bitmap and bitmap that will be subtracted
    static std::shared_ptr<BitmapSubtractor> Create( IBitmapPtr pSrcBitmap, IBitmapPtr pBitmapToSubtract );
    /// Creates an instance with pixel format of source bitmap (need to set it later) and bitmap that will be subtracted
    static std::shared_ptr<BitmapSubtractor> Create( PixelFormat srcPixelFormat, IBitmapPtr pBitmapToSubtract );
    /// Subtracts given bitmap from the source and returns result
    static IBitmapPtr Subtract( IBitmapPtr pSrcBitmap, IBitmapPtr pBitmapToSubtract );
};

ACMB_NAMESPACE_END