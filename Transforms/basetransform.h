#pragma once
#include "../Core/bitmap.h"

ACMB_NAMESPACE_BEGIN

class BaseTransform
{
protected:
    IBitmapPtr _pSrcBitmap;
    IBitmapPtr _pDstBitmap;

public:
    BaseTransform(IBitmapPtr pSrcBitmap);

    virtual void Run() = 0;
    IBitmapPtr RunAndGetBitmap();

    void SetSrcBitmap(IBitmapPtr pSrcBitmap);
};

ACMB_NAMESPACE_END
