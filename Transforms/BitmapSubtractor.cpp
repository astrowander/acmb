#include "BitmapSubtractor.h"
#include "../Core/camerasettings.h"
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

ACMB_NAMESPACE_BEGIN
template <PixelFormat pixelFormat>
class BitmapSubtractor_ final : public BitmapSubtractor
{
    using ChannelType = typename PixelFormatTraits<pixelFormat>::ChannelType;

public:

    BitmapSubtractor_( IBitmapPtr pSrcBitmap, const Settings& settings )
    : BitmapSubtractor( pSrcBitmap, settings )
    {
    }

    virtual void Run() override
    {
        auto pSrcBitmap = std::static_pointer_cast< Bitmap<pixelFormat> >( _pSrcBitmap );
        auto pBitmapToSubtract = std::static_pointer_cast< Bitmap<pixelFormat> >( _settings.pBitmapToSubtract );
        const int srcBlackLevel = pSrcBitmap->GetCameraSettings() ? pSrcBitmap->GetCameraSettings()->blackLevel : 0;        
        using ChannelType = typename PixelFormatTraits<pixelFormat>::ChannelType;

        oneapi::tbb::parallel_for( oneapi::tbb::blocked_range<int>( 0, _pSrcBitmap->GetHeight() ), [&] ( const oneapi::tbb::blocked_range<int>& range )
        {
            for ( int i = range.begin(); i < range.end(); ++i )
            {
                auto pSrcScanline = pSrcBitmap->GetScanline( i );
                auto pScanlineToSubtract = pBitmapToSubtract->GetScanline( i );                

                const size_t N = pSrcBitmap->GetWidth() * PixelFormatTraits<pixelFormat>::channelCount;

                for ( uint32_t j = 0; j < N; ++j )
                {
                    const auto srcVal = pSrcScanline[j];
                    const auto subtractVal = ( pScanlineToSubtract[j] - srcBlackLevel ) * _settings.multiplier;
                    //const auto res = ChannelType( std::min( srcBlackLevel + std::max( 0, srcVal - subtractVal ), maxChannel ) );
                    const auto res = ChannelType( std::max( float( srcBlackLevel ), ( srcVal - subtractVal ) ) );
                    pSrcScanline[j] = res;
                }
            }
        } );
        this->_pDstBitmap = this->_pSrcBitmap;
    }

    virtual void ValidateSettings() override
    {
        if ( !ArePixelFormatsCompatible( _pSrcBitmap->GetPixelFormat(), _settings.pBitmapToSubtract->GetPixelFormat() ) )
            throw std::invalid_argument( "bitmaps should have the same pixel format" );

        if ( _pSrcBitmap->GetWidth() != _settings.pBitmapToSubtract->GetWidth() || _pSrcBitmap->GetHeight() != _settings.pBitmapToSubtract->GetHeight() )
            throw std::invalid_argument( "bitmaps should have the same size" );
    }
};

BitmapSubtractor::BitmapSubtractor( IBitmapPtr pSrcBitmap, const Settings& settings )
: BaseTransform( pSrcBitmap )
, _settings(settings)
{
}

std::shared_ptr<BitmapSubtractor> BitmapSubtractor::Create( IBitmapPtr pSrcBitmap, const Settings& settings )
{
    if ( !pSrcBitmap )
        throw std::invalid_argument( "pSrcBitmap is null" );

    if ( !settings.pBitmapToSubtract )
        throw std::invalid_argument( "pBitmapToSubtract is null" );

    if ( !ArePixelFormatsCompatible( pSrcBitmap->GetPixelFormat(), settings.pBitmapToSubtract->GetPixelFormat() ) )
        throw std::invalid_argument( "bitmaps should have the same pixel format" );

    if ( pSrcBitmap->GetWidth() != settings.pBitmapToSubtract->GetWidth() || pSrcBitmap->GetHeight() != settings.pBitmapToSubtract->GetHeight() )
        throw std::invalid_argument( "bitmaps should have the same size" );

    switch ( pSrcBitmap->GetPixelFormat() )
    {
        case PixelFormat::Gray8:
            return std::make_shared<BitmapSubtractor_<PixelFormat::Gray8>>( pSrcBitmap, settings );
        case PixelFormat::Gray16:
            return std::make_shared<BitmapSubtractor_<PixelFormat::Gray16>>( pSrcBitmap, settings );
        case PixelFormat::Bayer16:
            return std::make_shared<BitmapSubtractor_<PixelFormat::Bayer16>>( pSrcBitmap, settings );
        case PixelFormat::RGB24:
            return std::make_shared<BitmapSubtractor_<PixelFormat::RGB24>>( pSrcBitmap, settings );
        case PixelFormat::RGB48:
            return std::make_shared<BitmapSubtractor_<PixelFormat::RGB48>>( pSrcBitmap, settings );
        default:
            throw std::runtime_error( "pixel format must be known" );
    }
}

std::shared_ptr<BitmapSubtractor> BitmapSubtractor::Create( PixelFormat srcPixelFormat, const Settings& settings )
{
    if ( !settings.pBitmapToSubtract )
        throw std::invalid_argument( "pBitmapToSubtract is null" );

    switch ( srcPixelFormat )
    {
        case PixelFormat::Gray8:
            return std::make_shared<BitmapSubtractor_<PixelFormat::Gray8>>( nullptr, settings );
        case PixelFormat::Gray16:
            return std::make_shared<BitmapSubtractor_<PixelFormat::Gray16>>( nullptr, settings );
        case PixelFormat::Bayer16:
            return std::make_shared<BitmapSubtractor_<PixelFormat::Bayer16>>( nullptr, settings );
        case PixelFormat::RGB24:
            return std::make_shared<BitmapSubtractor_<PixelFormat::RGB24>>( nullptr, settings );
        case PixelFormat::RGB48:
            return std::make_shared<BitmapSubtractor_<PixelFormat::RGB48>>( nullptr, settings );
        default:
            throw std::runtime_error( "pixel format must be known" );
    }
}

IBitmapPtr BitmapSubtractor::Subtract( IBitmapPtr pSrcBitmap, const Settings& settings )
{
    auto pSubtractor = Create( pSrcBitmap, settings );
    return pSubtractor->RunAndGetBitmap();
}

ACMB_NAMESPACE_END
