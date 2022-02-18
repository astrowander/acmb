#include "basetransform.h"


BaseTransform::BaseTransform(IBitmapPtr pSrcBitmap)
:_pSrcBitmap(pSrcBitmap)
{
}

IBitmapPtr BaseTransform::RunAndGetBitmap()
{
    Run();
    return _pDstBitmap;
}

void BaseTransform::SetSrcBitmap(IBitmapPtr pSrcBitmap)
{
    _pSrcBitmap = pSrcBitmap;
}
