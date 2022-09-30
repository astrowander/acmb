#pragma once
#include "basetransform.h"

ACMB_NAMESPACE_BEGIN

class BitmapSubtractor : public BaseTransform
{
public:
    using Settings = IBitmapPtr;
protected:
    IBitmapPtr _pBitmapToSubtract;
    BitmapSubtractor( IBitmapPtr pSrcBitmap, IBitmapPtr pBitmapToSubtract );

public:    
    static std::shared_ptr<BitmapSubtractor> Create( IBitmapPtr pSrcBitmap, IBitmapPtr pBitmapToSubtract );
    static std::shared_ptr<BitmapSubtractor> Create( PixelFormat srcPixelFormat, IBitmapPtr pBitmapToSubtract );
    static IBitmapPtr Subtract( IBitmapPtr pSrcBitmap, IBitmapPtr pBitmapToSubtract );
};

ACMB_NAMESPACE_END