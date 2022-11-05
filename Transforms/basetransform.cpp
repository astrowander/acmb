#include "basetransform.h"
ACMB_NAMESPACE_BEGIN

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
    if ( !pSrcBitmap )
        throw std::invalid_argument( "pSrcBitmap is null" );
    
    _pSrcBitmap = pSrcBitmap;
    ValidateSettings();
}

ACMB_NAMESPACE_END