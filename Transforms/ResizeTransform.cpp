#include "ResizeTransform.h"
#include "../Libs/avir/avir.h"

ACMB_NAMESPACE_BEGIN

template<PixelFormat pixelFormat>
class ResizeTransform_ : public ResizeTransform
{
    using ChannelType = typename PixelFormatTraits<pixelFormat>::ChannelType;
    static constexpr auto channelCount = PixelFormatTraits<pixelFormat>::channelCount;

public:
    ResizeTransform_( std::shared_ptr<IBitmap> pSrcBitmap, Size dstSize )
    : ResizeTransform( pSrcBitmap, dstSize )
    {    
    }

    virtual void Run() override
    {
        auto pSrcBitmap = std::static_pointer_cast< Bitmap<pixelFormat> >( _pSrcBitmap );

        _pDstBitmap = IBitmap::Create( _dstSize.width, _dstSize.height, pixelFormat );
        auto pDstBitmap = std::static_pointer_cast< Bitmap<pixelFormat> >( _pDstBitmap );

        avir::CImageResizer<> resizer( BitsPerChannel( pixelFormat ) );
        resizer.resizeImage( pSrcBitmap->GetScanline(0), pSrcBitmap->GetWidth(), pSrcBitmap->GetHeight(), 0, pDstBitmap->GetScanline( 0 ), _dstSize.width, _dstSize.height, channelCount, 0, nullptr );
    }

    virtual void ValidateSettings() override
    {
    }
};

ResizeTransform::ResizeTransform( std::shared_ptr<IBitmap> pSrcBitmap, Size dstSize )
    : BaseTransform( pSrcBitmap )
    , _dstSize( dstSize )
{
    if ( _dstSize.width == 0 || _dstSize.height == 0 )
        throw std::invalid_argument( "zero destination size" );
}

std::shared_ptr<ResizeTransform> ResizeTransform::Create( std::shared_ptr<IBitmap> pSrcBitmap, Size dstSize )
{
    if ( !pSrcBitmap )
        throw std::invalid_argument( "pSrcBitmap is null" );

    switch ( pSrcBitmap->GetPixelFormat() )
    {
        case PixelFormat::Gray8:
            return std::make_shared<ResizeTransform_<PixelFormat::Gray8>>( pSrcBitmap, dstSize );
        case PixelFormat::Gray16:
            return std::make_shared<ResizeTransform_<PixelFormat::Gray16>>( pSrcBitmap, dstSize );
        case PixelFormat::RGB24:
            return std::make_shared<ResizeTransform_<PixelFormat::RGB24>>( pSrcBitmap, dstSize );
        case PixelFormat::RGB48:
            return std::make_shared<ResizeTransform_<PixelFormat::RGB48>>( pSrcBitmap, dstSize );
        default:
            throw std::runtime_error( "unsupported pixel format" );
    }
}

std::shared_ptr<ResizeTransform> ResizeTransform::Create( PixelFormat pixelFormat, Size dstSize )
{
    switch ( pixelFormat )
    {
        case PixelFormat::Gray8:
            return std::make_shared<ResizeTransform_<PixelFormat::Gray8>>( nullptr, dstSize );
        case PixelFormat::Gray16:
            return std::make_shared<ResizeTransform_<PixelFormat::Gray16>>( nullptr, dstSize );
        case PixelFormat::RGB24:
            return std::make_shared<ResizeTransform_<PixelFormat::RGB24>>( nullptr, dstSize );
        case PixelFormat::RGB48:
            return std::make_shared<ResizeTransform_<PixelFormat::RGB48>>( nullptr, dstSize );
        default:
            throw std::runtime_error( "unsupported pixel format" );
    }
}

IBitmapPtr ResizeTransform::Resize( IBitmapPtr pSrcBitmap, Size dstSize )
{
    auto pResizeTransform = Create( pSrcBitmap, dstSize );
    return pResizeTransform->RunAndGetBitmap();
}

void ResizeTransform::CalcParams( std::shared_ptr<ImageParams> pParams )
{
    _width = _dstSize.width;
    _height = _dstSize.height;
    _pixelFormat = pParams->GetPixelFormat();
}
ACMB_NAMESPACE_END
