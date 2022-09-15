#include "BitmapSubtractor.h"
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

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

IBitmapPtr BaseBitmapSubtractor::Subtract( IBitmapPtr pSrcBitmap, IBitmapPtr pBitmapToSubtract )
{
    auto pSubtractor = Create( pSrcBitmap, pBitmapToSubtract );
    return pSubtractor->RunAndGetBitmap();
}

template<PixelFormat pixelFormat>
BitmapSubtractor<pixelFormat>::BitmapSubtractor( IBitmapPtr pSrcBitmap, IBitmapPtr pBitmapToSubtract )
: BaseBitmapSubtractor( pSrcBitmap, pBitmapToSubtract )
{
}

template<PixelFormat pixelFormat>
void BitmapSubtractor<pixelFormat>::Run()
{
    auto pSrcBitmap = std::static_pointer_cast< Bitmap<pixelFormat> >( _pSrcBitmap );
    auto pBitmapToSubtract = std::static_pointer_cast< Bitmap<pixelFormat> >( _pBitmapToSubtract );

    oneapi::tbb::parallel_for( oneapi::tbb::blocked_range<int>( 0, _pSrcBitmap->GetHeight() ), [pSrcBitmap, pBitmapToSubtract] ( const oneapi::tbb::blocked_range<int>& range )
    {
        for ( int i = range.begin(); i < range.end(); ++i )
        {
            auto pSrcScanline = pSrcBitmap->GetScanline( i );
            auto pScanlineToSubtract = pBitmapToSubtract->GetScanline( i );

            const size_t N = pSrcBitmap->GetWidth() * PixelFormatTraits<pixelFormat>::channelCount;

            for ( uint32_t j = 0; j < N; ++j )
            {
                pSrcScanline[j] = std::max( 0, pSrcScanline[j] - pScanlineToSubtract[j] );
            }
        }
    } );
    this->_pDstBitmap = this->_pSrcBitmap;
}
