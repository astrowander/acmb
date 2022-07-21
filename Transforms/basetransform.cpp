#include "basetransform.h"


BaseTransform::BaseTransform(IBitmapPtr pSrcBitmap)
:_pSrcBitmap(pSrcBitmap)
{
    if ( !pSrcBitmap )
        throw std::invalid_argument( "pSrcBitmap is null" );
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
