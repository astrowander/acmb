#include "AffineTransform.h"
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
using namespace oneapi::tbb;

ACMB_NAMESPACE_BEGIN

template<PixelFormat pixelFormat>
class AffineTransformImpl : public AffineTransform
{
    using ChannelType = typename PixelFormatTraits<pixelFormat>::ChannelType;
    static constexpr auto channelCount = PixelFormatTraits<pixelFormat>::channelCount;

public:
    AffineTransformImpl( std::shared_ptr<Bitmap<pixelFormat>> pSrcBitmap, const Settings& settings )
    : AffineTransform( pSrcBitmap, settings )
    {
        if ( !settings.pBgColor )
        {
            _settings.pBgColor = IColor::Create( pixelFormat, { 0, 0, 0, 0 } );
        }
    }

    virtual void Run() override
    {
        const auto width = _pSrcBitmap->GetWidth();
        const auto height = _pSrcBitmap->GetHeight();

        auto pSrcBitmap = std::static_pointer_cast< Bitmap<pixelFormat> >(_pSrcBitmap);
        auto pDstBitmap = std::make_shared<Bitmap<pixelFormat>>( _pSrcBitmap->GetWidth(), _pSrcBitmap->GetHeight() );

        const auto invTransform = _settings.transform.invert();
        parallel_for( blocked_range<uint32_t>( 0, height ), [&] ( const blocked_range<uint32_t>& r )
        {
            for ( uint32_t y = r.begin(); y < r.end(); ++y )            
            {
                auto pDstPixel = pDstBitmap->GetScanline( y );
                for ( uint32_t x = 0; x < width; ++x )
                {
                    PointD p{ double( x ), double( y ) };
                    invTransform.transform( &p.x, &p.y );

                    for ( uint32_t ch = 0; ch < channelCount; ++ch )
                        pDstPixel[ch] = pSrcBitmap->GetInterpolatedChannel( p.x, p.y, ch );

                    pDstPixel += channelCount;
                }                
            }
        } );

        _pDstBitmap = pDstBitmap;
    }

    virtual void ValidateSettings() override
    {
    }
};

AffineTransform::AffineTransform( IBitmapPtr pSrcBitmap, const Settings& controls )
    : BaseTransform( pSrcBitmap )
    , _settings( controls )
{}

std::shared_ptr<AffineTransform> AffineTransform::Create( IBitmapPtr pSrcBitmap, const Settings& controls )
{
    switch ( pSrcBitmap->GetPixelFormat() )
    {
        case PixelFormat::RGB24:
            return std::make_shared<AffineTransformImpl<PixelFormat::RGB24>>( std::static_pointer_cast< Bitmap<PixelFormat::RGB24> >(pSrcBitmap), controls );
        case PixelFormat::RGB48:
            return std::make_shared<AffineTransformImpl<PixelFormat::RGB48>>( std::static_pointer_cast< Bitmap<PixelFormat::RGB48> >(pSrcBitmap), controls );
        case PixelFormat::Gray8:
            return std::make_shared<AffineTransformImpl<PixelFormat::Gray8>>( std::static_pointer_cast< Bitmap<PixelFormat::Gray8> >(pSrcBitmap), controls );
        case PixelFormat::Gray16:
            return std::make_shared<AffineTransformImpl<PixelFormat::Gray16>>( std::static_pointer_cast< Bitmap<PixelFormat::Gray16> >(pSrcBitmap), controls );

        default:
            throw std::invalid_argument( "AffineTransform: Unsupported pixel format" );
    }
}

std::shared_ptr<AffineTransform> AffineTransform::Create( PixelFormat pixelFormat, const Settings& controls )
{
    switch ( pixelFormat )
    {
        case PixelFormat::RGB24:
            return std::make_shared<AffineTransformImpl<PixelFormat::RGB24>>( nullptr, controls );
        case PixelFormat::RGB48:
            return std::make_shared<AffineTransformImpl<PixelFormat::RGB48>>( nullptr, controls );
        case PixelFormat::Gray8:
            return std::make_shared<AffineTransformImpl<PixelFormat::Gray8>>( nullptr, controls );
        case PixelFormat::Gray16:
            return std::make_shared<AffineTransformImpl<PixelFormat::Gray16>>( nullptr, controls );

        default:
            throw std::invalid_argument( "AffineTransform: Unsupported pixel format" );
    }
}

IBitmapPtr AffineTransform::ApplyTransform( IBitmapPtr pSrcBitmap, const Settings& controls )
{
    auto pAffineTransform = Create( pSrcBitmap, controls );
    return pAffineTransform->RunAndGetBitmap();
}

ACMB_NAMESPACE_END