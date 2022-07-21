#include "BitmapSubtractor.h"

BaseBitmapSubtractor::BaseBitmapSubtractor( IBitmapPtr pSrcBitmap, IBitmapPtr pBitmapToSubtract )
: BaseTransform( pSrcBitmap )
, _pBitmapToSubtract(pBitmapToSubtract)
{
    if ( !pBitmapToSubtract )
        throw std::invalid_argument( "pBitmapToSubtract is null" );

    if (pSrcBitmap->GetPixelFormat() != pBitmapToSubtract->GetPixelFormat() )
        throw std::invalid_argument( "bitmaps should have the same pixel format" );

    if ( pSrcBitmap->GetWidth() != pBitmapToSubtract->GetWidth() || pSrcBitmap->GetHeight() != pBitmapToSubtract->GetHeight() )
        throw std::invalid_argument( "bitmaps should have the same size" );
}

std::shared_ptr<BaseBitmapSubtractor> BaseBitmapSubtractor::Create( IBitmapPtr pSrcBitmap, IBitmapPtr pBitmapToSubtract )
{
    switch ( pSrcBitmap->GetPixelFormat() )
    {
        case PixelFormat::Gray8:
            return std::make_shared<BitmapSubtractor<PixelFormat::Gray8>>( pSrcBitmap, pBitmapToSubtract );
        case PixelFormat::Gray16:
            return std::make_shared<BitmapSubtractor<PixelFormat::Gray16>>( pSrcBitmap, pBitmapToSubtract );
        case PixelFormat::RGB24:
            return std::make_shared<BitmapSubtractor<PixelFormat::RGB24>>( pSrcBitmap, pBitmapToSubtract );
        case PixelFormat::RGB48:
            return std::make_shared<BitmapSubtractor<PixelFormat::RGB48>>( pSrcBitmap, pBitmapToSubtract );
        default:
            throw std::runtime_error( "pixel format must be known" );
    }
}