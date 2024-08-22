#include "CropTransform.h"
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <cstring>
ACMB_NAMESPACE_BEGIN

template<PixelFormat pixelFormat>
class CropTransform_ : public CropTransform
{
    using ChannelType = typename PixelFormatTraits<pixelFormat>::ChannelType;
    static constexpr auto channelCount = PixelFormatTraits<pixelFormat>::channelCount;

public:
    CropTransform_( std::shared_ptr<IBitmap> pSrcBitmap, Rect dstRect )
        : CropTransform( pSrcBitmap, dstRect )
    {
    }

    virtual void Run() override
    {
        auto pSrcBitmap = std::static_pointer_cast< Bitmap<pixelFormat> >( _pSrcBitmap );

        _pDstBitmap = IBitmap::Create( _dstRect.width, _dstRect.height, pixelFormat );
        auto pDstBitmap = std::static_pointer_cast< Bitmap<pixelFormat> >( _pDstBitmap );

        oneapi::tbb::parallel_for( oneapi::tbb::blocked_range<int>( 0, _dstRect.height ), [&] ( const oneapi::tbb::blocked_range<int>& range )
        {
            for ( int i = range.begin(); i < range.end(); ++i )
            {
                memcpy( pDstBitmap->GetScanline( i ), pSrcBitmap->GetScanline( i + _dstRect.y ) + _dstRect.x * channelCount, _dstRect.width * BytesPerPixel( pixelFormat ) );
            }
        } );
    }

    virtual void ValidateSettings() override
    {
        if ( _dstRect.x + _dstRect.width > int ( _pSrcBitmap->GetWidth() ) ||
             _dstRect.y + _dstRect.height > int ( _pSrcBitmap->GetHeight() ) )
        {
            throw std::runtime_error( "crop rect exceeds the source bitmap" );
        }
    }
};

CropTransform::CropTransform( std::shared_ptr<IBitmap> pSrcBitmap, Rect dstRect )
    : BaseTransform( pSrcBitmap )
    , _dstRect( dstRect )
{
    if ( _dstRect.x < 0 || _dstRect.y < 0 )
        throw std::invalid_argument( "invalid destination point" );

    if ( _dstRect.width <= 0 || _dstRect.height <= 0 )
        throw std::invalid_argument( "invalid destination size" );   
}

std::shared_ptr<CropTransform> CropTransform::Create( std::shared_ptr<IBitmap> pSrcBitmap, Rect dstRect )
{
    if ( !pSrcBitmap )
        throw std::invalid_argument( "pSrcBitmap is null" );

    if ( dstRect.x + dstRect.width > int( pSrcBitmap->GetWidth() ) ||
         dstRect.y + dstRect.height > int( pSrcBitmap->GetHeight() ) )
    {
        throw std::runtime_error( "crop rect exceeds the source bitmap" );
    }

    switch ( pSrcBitmap->GetPixelFormat() )
    {
        case PixelFormat::Gray8:
            return std::make_shared<CropTransform_<PixelFormat::Gray8>>( pSrcBitmap, dstRect );
        case PixelFormat::Gray16:
            return std::make_shared<CropTransform_<PixelFormat::Gray16>>( pSrcBitmap, dstRect );
        case PixelFormat::RGB24:
            return std::make_shared<CropTransform_<PixelFormat::RGB24>>( pSrcBitmap, dstRect );
        case PixelFormat::RGB48:
            return std::make_shared<CropTransform_<PixelFormat::RGB48>>( pSrcBitmap, dstRect );
        default:
            throw std::runtime_error( "unsupported pixel format" );
    }
}

std::shared_ptr<CropTransform> CropTransform::Create( PixelFormat pixelFormat, Rect dstRect )
{
    switch ( pixelFormat )
    {
        case PixelFormat::Gray8:
            return std::make_shared<CropTransform_<PixelFormat::Gray8>>( nullptr, dstRect );
        case PixelFormat::Gray16:
            return std::make_shared<CropTransform_<PixelFormat::Gray16>>( nullptr, dstRect );
        case PixelFormat::RGB24:
            return std::make_shared<CropTransform_<PixelFormat::RGB24>>( nullptr, dstRect );
        case PixelFormat::RGB48:
            return std::make_shared<CropTransform_<PixelFormat::RGB48>>( nullptr, dstRect );
        default:
            throw std::runtime_error( "unsupported pixel format" );
    }
}

IBitmapPtr CropTransform::Crop( IBitmapPtr pSrcBitmap, Rect dstRect )
{
    auto pCropTransform = Create( pSrcBitmap, dstRect );
    return pCropTransform->RunAndGetBitmap();
}

IBitmapPtr CropTransform::CropAndFill( IBitmapPtr pSrcBitmap, Rect roi, IColorPtr pColor )
{
    if ( !pSrcBitmap )
        throw std::invalid_argument( "pSrcBitmap is null" );

    if ( !pColor )
        throw std::invalid_argument( "pColor is null" );

    if ( pSrcBitmap->GetPixelFormat() != pColor->GetPixelFormat() )
        throw std::invalid_argument( "pixel formats do not match" );

    auto pRes = IBitmap::Create( roi.width, roi.height, pColor );

    // area on original image that will be copied
    Rect srcRect
    {
        .x = roi.x < 0 ? 0 : roi.x,
        .y = roi.y < 0 ? 0 : roi.y
    };

    srcRect.width = std::min( roi.GetRight(), int32_t( pSrcBitmap->GetWidth() ) ) - srcRect.x;
    srcRect.height = std::min( roi.GetBottom(), int32_t( pSrcBitmap->GetHeight() ) ) - srcRect.y;
    if ( srcRect.width <= 0 || srcRect.height <= 0 )
        return pRes;

    const uint32_t bypp = BytesPerPixel( pSrcBitmap->GetPixelFormat() );

    Point originPoint { .x = roi.x < 0 ? -roi.x : 0, .y = roi.y < 0 ? -roi.y : 0 };
    oneapi::tbb::parallel_for( oneapi::tbb::blocked_range<int>( 0, srcRect.height ), [&] ( const oneapi::tbb::blocked_range<int>& range )
    {
        for ( int i = range.begin(); i < range.end(); ++i )
        {
            auto pSrcScanline = pSrcBitmap->GetPlanarScanline( i + srcRect.y );
            auto pResScanline = pRes->GetPlanarScanline( i + originPoint.y );
            memcpy( pResScanline + originPoint.x * bypp, pSrcScanline + bypp * srcRect.x, srcRect.width * bypp );
        }
    });

    return pRes;
}

void CropTransform::CalcParams( std::shared_ptr<ImageParams> pParams )
{
    _width = _dstRect.width;
    _height = _dstRect.height;
    _pixelFormat = pParams->GetPixelFormat();
}

ACMB_NAMESPACE_END
