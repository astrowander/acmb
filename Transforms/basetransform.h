#ifndef BASETRANSFORM_H
#define BASETRANSFORM_H
#include "../Core/bitmap.h"

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

#endif // BASETRANSFORM_H
