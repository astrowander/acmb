#include "CenterObjectTransform.h"
#include "converter.h"
#include "CropTransform.h"

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/enumerable_thread_specific.h>

#include <algorithm>
#include <execution>

#undef min
#undef max

ACMB_NAMESPACE_BEGIN

template<PixelFormat pixelFormat>
class CenterObjectTransform_ : public CenterObjectTransform
{
    using ChannelType = typename PixelFormatTraits<pixelFormat>::ChannelType;
    static constexpr auto channelCount = PixelFormatTraits<pixelFormat>::channelCount;

    static constexpr PixelFormat cGrayFormat = ConstructPixelFormat( BitsPerChannel( pixelFormat ), 1 );

public:
    CenterObjectTransform_( std::shared_ptr<IBitmap> pSrcBitmap, const CenterObjectTransform::Settings& settings )
    : CenterObjectTransform( pSrcBitmap, settings )
    {}

    virtual void Run() override
    {
        auto pSrcBitmap = std::static_pointer_cast< Bitmap<pixelFormat> >( _pSrcBitmap );
        
        std::shared_ptr<Bitmap<cGrayFormat>> pGrayBitmap;
        if constexpr ( cGrayFormat == pixelFormat )
        {
            pGrayBitmap = static_pointer_cast< Bitmap<cGrayFormat> >( pSrcBitmap->Clone() );
        }
        else        
        {
            auto pConverter = Converter::Create( pSrcBitmap, cGrayFormat );
            pGrayBitmap = static_pointer_cast<Bitmap<cGrayFormat>>( pConverter->RunAndGetBitmap() );
        }

        uint32_t bgAreaSize = 0.15f * std::max( pSrcBitmap->GetWidth(), pSrcBitmap->GetHeight() );
        uint32_t srcPixelCount = bgAreaSize * bgAreaSize;
        std::vector<ChannelType> dataCopy( srcPixelCount );
        oneapi::tbb::parallel_for( oneapi::tbb::blocked_range<uint32_t>( 0, bgAreaSize ), [&] ( const oneapi::tbb::blocked_range<uint32_t>& range )
        {
            for ( uint32_t i = range.begin(); i < range.end(); ++i )
            {
                memcpy( dataCopy.data() + i * bgAreaSize, pGrayBitmap->GetScanline( i ), bgAreaSize * sizeof( ChannelType ) );
            }
        } );

        ChannelType* median = dataCopy.data() + srcPixelCount / 2;
        std::nth_element( std::execution::par, dataCopy.data(), median, dataCopy.data() + srcPixelCount );

        const ChannelType threshold = ChannelType( std::min( *median * (1 + settings_.threshold / 100.0f), float( PixelFormatTraits<pixelFormat>::channelMax ) ) );

        struct DataPerThread
        {
            float totalLuminance = 0;
            PointF center;
        };

        oneapi::tbb::enumerable_thread_specific<DataPerThread> dataPerThread;
        oneapi::tbb::parallel_for( oneapi::tbb::blocked_range<uint32_t>( 0, pSrcBitmap->GetHeight() ), [&] ( const oneapi::tbb::blocked_range<uint32_t>& range )
        {
            auto& dataLocal = dataPerThread.local();
            for ( uint32_t y = range.begin(); y < range.end(); ++y )
            {
                ChannelType* pPixel = pGrayBitmap->GetScanline( y );
                for ( uint32_t x = 0; x < pSrcBitmap->GetWidth(); ++x )
                {
                    auto value = *pPixel++;
                    if ( value < threshold )
                        continue;

                    const auto luminance = value - threshold;
                    dataLocal.totalLuminance += luminance;
                    dataLocal.center.x += x * luminance;
                    dataLocal.center.y += y * luminance;
                }
            }
        } );

        float totalLuminance = 0;
        PointF center;
        for ( const auto& data : dataPerThread )
        {
            totalLuminance += data.totalLuminance;
            center.x += data.center.x;
            center.y += data.center.y;
        }

        center.x /= totalLuminance;
        center.y /= totalLuminance;

        Rect dstRect { .width = settings_.dstSize.width, .height = settings_.dstSize.height };
        dstRect.x = int( center.x - settings_.dstSize.width / 2 );
        dstRect.y = int( center.y - settings_.dstSize.height / 2 );

        //if ( dstRect.x < 0 || dstRect.y < 0 || dstRect.x + dstRect.width > pSrcBitmap->GetWidth() || dstRect.y + dstRect.height > pSrcBitmap->GetHeight() )
           // return;

        std::array<uint32_t,4> channels = { *median, *median, *median, *median };
        _pDstBitmap = CropTransform::CropAndFill( pSrcBitmap, dstRect, IColor::Create( pixelFormat, channels ) );
    }

    virtual void ValidateSettings() override
    {
        if ( settings_.threshold < 0 )
            throw std::invalid_argument( "threshold must be >= 0" );
    }
};

CenterObjectTransform::CenterObjectTransform( IBitmapPtr pSrcBitmap, const CenterObjectTransform::Settings& settings )
: BaseTransform( pSrcBitmap )
, settings_( settings )
{
    if ( settings.threshold < 0 )
        throw std::invalid_argument( "threshold must be >= 0" );
}

std::shared_ptr<CenterObjectTransform> CenterObjectTransform::Create( IBitmapPtr pSrcBitmap, const CenterObjectTransform::Settings& settings )
{
    if ( !pSrcBitmap )
        throw std::invalid_argument( "pSrcBitmap is null" );

    if ( settings.threshold < 0 )
        throw std::invalid_argument( "threshold must be >= 0" );

    switch ( pSrcBitmap->GetPixelFormat() )
    {
        case PixelFormat::Gray8:
            return std::make_shared<CenterObjectTransform_<PixelFormat::Gray8>>( pSrcBitmap, settings );
        case PixelFormat::Gray16:
            return std::make_shared<CenterObjectTransform_<PixelFormat::Gray16>>( pSrcBitmap, settings );
        case PixelFormat::RGB24:
            return std::make_shared<CenterObjectTransform_<PixelFormat::RGB24>>( pSrcBitmap, settings );
        case PixelFormat::RGB48:
            return std::make_shared<CenterObjectTransform_<PixelFormat::RGB48>>( pSrcBitmap, settings );
        default:
            throw std::runtime_error( "unsupported pixel format" );
    }
}

std::shared_ptr<CenterObjectTransform> CenterObjectTransform::Create( PixelFormat pixelFormat, const CenterObjectTransform::Settings& settings )
{
    if ( settings.threshold < 0 )
        throw std::invalid_argument( "threshold must be >= 0" );

    switch ( pixelFormat )
    {
        case PixelFormat::Gray8:
            return std::make_shared<CenterObjectTransform_<PixelFormat::Gray8>>( nullptr, settings );
        case PixelFormat::Gray16:
            return std::make_shared<CenterObjectTransform_<PixelFormat::Gray16>>( nullptr, settings );
        case PixelFormat::RGB24:
            return std::make_shared<CenterObjectTransform_<PixelFormat::RGB24>>( nullptr, settings );
        case PixelFormat::RGB48:
            return std::make_shared<CenterObjectTransform_<PixelFormat::RGB48>>( nullptr, settings );
        default:
            throw std::runtime_error( "unsupported pixel format" );
    }
}

IBitmapPtr CenterObjectTransform::CenterObject( IBitmapPtr pSrcBitmap, const CenterObjectTransform::Settings& settings )
{
    auto pCenterTransform = Create( pSrcBitmap, settings );
    return pCenterTransform->RunAndGetBitmap();
}

void CenterObjectTransform::CalcParams( std::shared_ptr<ImageParams> pParams )
{
    _width = settings_.dstSize.width;
    _height = settings_.dstSize.height;
    _pixelFormat = pParams->GetPixelFormat();
}

ACMB_NAMESPACE_END