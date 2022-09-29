#include "BitmapSubtractor.h"
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

ACMB_NAMESPACE_BEGIN
template <PixelFormat pixelFormat>
class BitmapSubtractor_ final : public BitmapSubtractor
{
    using ChannelType = typename PixelFormatTraits<pixelFormat>::ChannelType;

public:

    BitmapSubtractor_( IBitmapPtr pSrcBitmap, IBitmapPtr pBitmapToSubtract )
    : BitmapSubtractor( pSrcBitmap, pBitmapToSubtract )
    {
    }

    virtual void Run() override
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
};

BitmapSubtractor::BitmapSubtractor( IBitmapPtr pSrcBitmap, IBitmapPtr pBitmapToSubtract )
: BaseTransform( pSrcBitmap )
, _pBitmapToSubtract(pBitmapToSubtract)
{
}

std::shared_ptr<BitmapSubtractor> BitmapSubtractor::Create( IBitmapPtr pSrcBitmap, IBitmapPtr pBitmapToSubtract )
{
    if ( !pSrcBitmap )
        throw std::invalid_argument( "pSrcBitmap is null" );

    if ( !pBitmapToSubtract )
        throw std::invalid_argument( "pBitmapToSubtract is null" );

    if ( pSrcBitmap->GetPixelFormat() != pBitmapToSubtract->GetPixelFormat() )
        throw std::invalid_argument( "bitmaps should have the same pixel format" );

    if ( pSrcBitmap->GetWidth() != pBitmapToSubtract->GetWidth() || pSrcBitmap->GetHeight() != pBitmapToSubtract->GetHeight() )
        throw std::invalid_argument( "bitmaps should have the same size" );

    switch ( pSrcBitmap->GetPixelFormat() )
    {
        case PixelFormat::Gray8:
            return std::make_shared<BitmapSubtractor_<PixelFormat::Gray8>>( pSrcBitmap, pBitmapToSubtract );
        case PixelFormat::Gray16:
            return std::make_shared<BitmapSubtractor_<PixelFormat::Gray16>>( pSrcBitmap, pBitmapToSubtract );
        case PixelFormat::RGB24:
            return std::make_shared<BitmapSubtractor_<PixelFormat::RGB24>>( pSrcBitmap, pBitmapToSubtract );
        case PixelFormat::RGB48:
            return std::make_shared<BitmapSubtractor_<PixelFormat::RGB48>>( pSrcBitmap, pBitmapToSubtract );
        default:
            throw std::runtime_error( "pixel format must be known" );
    }
}

std::shared_ptr<BitmapSubtractor> BitmapSubtractor::Create( PixelFormat srcPixelFormat, IBitmapPtr pBitmapToSubtract )
{
    if ( !pBitmapToSubtract )
        throw std::invalid_argument( "pBitmapToSubtract is null" );

    switch ( srcPixelFormat )
    {
        case PixelFormat::Gray8:
            return std::make_shared<BitmapSubtractor_<PixelFormat::Gray8>>( nullptr, pBitmapToSubtract );
        case PixelFormat::Gray16:
            return std::make_shared<BitmapSubtractor_<PixelFormat::Gray16>>( nullptr, pBitmapToSubtract );
        case PixelFormat::RGB24:
            return std::make_shared<BitmapSubtractor_<PixelFormat::RGB24>>( nullptr, pBitmapToSubtract );
        case PixelFormat::RGB48:
            return std::make_shared<BitmapSubtractor_<PixelFormat::RGB48>>( nullptr, pBitmapToSubtract );
        default:
            throw std::runtime_error( "pixel format must be known" );
    }
}

IBitmapPtr BitmapSubtractor::Subtract( IBitmapPtr pSrcBitmap, IBitmapPtr pBitmapToSubtract )
{
    auto pSubtractor = Create( pSrcBitmap, pBitmapToSubtract );
    return pSubtractor->RunAndGetBitmap();
}

ACMB_NAMESPACE_END