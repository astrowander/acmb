#include "basetransform.h"
ACMB_NAMESPACE_BEGIN

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
    if ( !pSrcBitmap )
        throw std::invalid_argument( "pSrcBitmap is null" );

    //if ( pSrcBitmap->GetPixelFormat() != _pixelFormat )
      //  throw std::invalid_argument( "pSrcBitmap has different pixel format" );

    _pSrcBitmap = pSrcBitmap;
}

ACMB_NAMESPACE_END