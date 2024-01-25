#include "SaturationTransform.h"
#include "./../Tools/mathtools.h"

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

ACMB_NAMESPACE_BEGIN

template<PixelFormat pixelFormat>
class SaturationTransform_ final : public SaturationTransform
{
    using ChannelType = typename PixelFormatTraits<pixelFormat>::ChannelType;
public:
    SaturationTransform_( IBitmapPtr pSrcBitmap, const Settings& settings )
        : SaturationTransform( pSrcBitmap, settings )
    {}

    virtual void Run() override
    {
        if ( _intensity == 1.0f )
        {
            this->_pDstBitmap = this->_pSrcBitmap;
            return;
        }

        auto pSrcBitmap = std::static_pointer_cast< Bitmap<pixelFormat> >(_pSrcBitmap);
        //const int maxChannel = pSrcBitmap->GetCameraSettings() ? pSrcBitmap->GetCameraSettings()->maxChannel : PixelFormatTraits<pixelFormat>::channelMax;
        oneapi::tbb::parallel_for( oneapi::tbb::blocked_range<uint32_t>( 0u, _pSrcBitmap->GetHeight() ), [&] ( const oneapi::tbb::blocked_range<uint32_t>& range )
        {
            for ( uint32_t i = range.begin(); i < range.end(); ++i )
            {
                auto pSrcScanline = pSrcBitmap->GetScanline( i );
                for ( uint32_t j = 0; j < pSrcBitmap->GetWidth(); ++j )
                {
                    auto rgb = std::span<ChannelType, 3>( pSrcBitmap->GetScanline( i ) + j * 3, 3 );
                    auto hsl = RgbToHsl( rgb );
                    hsl[1] *= _intensity;
                    HslToRgb( hsl, rgb );
                }
            }
        } );

        this->_pDstBitmap = this->_pSrcBitmap;
    }
    
    virtual void ValidateSettings() override
    {
        if ( GetColorSpace( _pSrcBitmap->GetPixelFormat() ) != ColorSpace::RGB )
            throw std::invalid_argument( "unsupported pixel format" );
    }
};

SaturationTransform::SaturationTransform( IBitmapPtr pSrcBitmap, float intensity )
: BaseTransform( pSrcBitmap )
, _intensity( std::clamp( intensity, 0.0f, 4.0f ) )
{
}

std::shared_ptr<SaturationTransform> SaturationTransform::Create( IBitmapPtr pSrcBitmap, float intensity )
{
    if ( !pSrcBitmap )
        throw std::invalid_argument( "pSrcBitmap is null" );

    if ( GetColorSpace( pSrcBitmap->GetPixelFormat() ) != ColorSpace::RGB )
        throw std::invalid_argument( "unsupported pixel format" );

    switch ( pSrcBitmap->GetPixelFormat() )
    {
        case PixelFormat::RGB24:
            return std::make_shared<SaturationTransform_<PixelFormat::RGB24>>( pSrcBitmap, intensity );
        case PixelFormat::RGB48:
            return std::make_shared<SaturationTransform_<PixelFormat::RGB48>>( pSrcBitmap, intensity );
        default:
            throw std::invalid_argument( "unsupported pixel format" );
    }
}

std::shared_ptr<SaturationTransform> SaturationTransform::Create( PixelFormat pixelFormat, float intensity )
{
    switch ( pixelFormat )
    {
        case PixelFormat::RGB24:
            return std::make_shared<SaturationTransform_<PixelFormat::RGB24>>( nullptr, intensity );
        case PixelFormat::RGB48:
            return std::make_shared<SaturationTransform_<PixelFormat::RGB48>>( nullptr, intensity );
        default:
            throw std::invalid_argument( "unsupported pixel format" );
    }
}

IBitmapPtr SaturationTransform::Saturate( IBitmapPtr srcBitmap, float intensity )
{
    auto pTransform = SaturationTransform::Create( srcBitmap, intensity );
    return pTransform->RunAndGetBitmap();
}

ACMB_NAMESPACE_END