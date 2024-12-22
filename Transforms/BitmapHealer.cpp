#include "BitmapHealer.h"
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
using namespace oneapi::tbb;

ACMB_NAMESPACE_BEGIN

template<PixelFormat pixelFormat>
class BitmapHealerImpl : public BitmapHealer
{
    using ChannelType = typename PixelFormatTraits<pixelFormat>::ChannelType;
    static constexpr auto channelCount = PixelFormatTraits<pixelFormat>::channelCount;
    static constexpr auto channelMax = PixelFormatTraits<pixelFormat>::channelMax;

public:
    BitmapHealerImpl( std::shared_ptr<Bitmap<pixelFormat>> pSrcBitmap, const Settings& settings )
    : BitmapHealer( pSrcBitmap, settings )
    {
    }

    void ApplyPatch( const Patch& patch )
    {
        std::vector<float> coeffs( patch.radius + 1 );
        for ( int i = 0; i < patch.radius; ++i )
            coeffs[i] = pow( float ( i + 1 ) / ( patch.radius + 1 ), patch.gamma );
        
        coeffs[patch.radius] = 1.0f;

        auto pSrcBitmap = std::static_pointer_cast< Bitmap<pixelFormat> >( _pSrcBitmap );

        for ( int dy = -patch.radius; dy <= patch.radius; ++dy )
        {
            for ( int dx = -patch.radius; dx <= patch.radius; ++dx )
            {
                Point srcPos = patch.from + Point( dx, dy );
                if ( srcPos.x < 0 || srcPos.y < 0 || srcPos.x >= int( pSrcBitmap->GetWidth() ) || srcPos.y >= int( pSrcBitmap->GetHeight() ) )
                    continue;

                Point dstPos = patch.to + Point( dx, dy );

                if ( dstPos.x < 0 || dstPos.y < 0 || dstPos.x >= int( pSrcBitmap->GetWidth() ) || dstPos.y >= int( pSrcBitmap->GetHeight() ) )
                    continue;

                int layer = patch.radius - std::max( std::abs( dx ), std::abs( dy ) );

                const ChannelType* pSrcScanline = pSrcBitmap->GetScanline( srcPos.y );
                ChannelType* pDstScanline = pSrcBitmap->GetScanline( dstPos.y );

                for ( int i = 0; i < channelCount; ++i )
                {
                    pDstScanline[dstPos.x * channelCount + i] = 
                        ChannelType( std::clamp<float>( pSrcScanline[srcPos.x * channelCount + i] * coeffs[layer] + 
                        pDstScanline[dstPos.x * channelCount + i] * ( 1 - coeffs[layer] ), 0, channelMax ) );
                }

            }
        }
    }

    virtual void Run() override
    {
        for ( const auto& patch : _patches )
            ApplyPatch( patch );

        _pDstBitmap = _pSrcBitmap;
    }

    virtual void ValidateSettings() override
    {
    }
};

BitmapHealer::BitmapHealer( IBitmapPtr pSrcBitmap, const Settings& settings )
: BaseTransform( pSrcBitmap )
, _patches( settings )
{
}

std::shared_ptr<BitmapHealer> BitmapHealer::Create( IBitmapPtr pSrcBitmap, const Settings& controls )
{
    switch ( pSrcBitmap->GetPixelFormat() )
    {
        case PixelFormat::RGB24:
            return std::make_shared<BitmapHealerImpl<PixelFormat::RGB24>>( std::static_pointer_cast< Bitmap<PixelFormat::RGB24> >(pSrcBitmap), controls );
        case PixelFormat::RGB48:
            return std::make_shared<BitmapHealerImpl<PixelFormat::RGB48>>( std::static_pointer_cast< Bitmap<PixelFormat::RGB48> >(pSrcBitmap), controls );
        case PixelFormat::Gray8:
            return std::make_shared<BitmapHealerImpl<PixelFormat::Gray8>>( std::static_pointer_cast< Bitmap<PixelFormat::Gray8> >(pSrcBitmap), controls );
        case PixelFormat::Gray16:
            return std::make_shared<BitmapHealerImpl<PixelFormat::Gray16>>( std::static_pointer_cast< Bitmap<PixelFormat::Gray16> >(pSrcBitmap), controls );

        default:
            throw std::invalid_argument( "BitmapHealer: Unsupported pixel format" );
    }
}

std::shared_ptr<BitmapHealer> BitmapHealer::Create( PixelFormat pixelFormat, const Settings& controls )
{
    switch ( pixelFormat )
    {
        case PixelFormat::RGB24:
            return std::make_shared<BitmapHealerImpl<PixelFormat::RGB24>>( nullptr, controls );
        case PixelFormat::RGB48:
            return std::make_shared<BitmapHealerImpl<PixelFormat::RGB48>>( nullptr, controls );
        case PixelFormat::Gray8:
            return std::make_shared<BitmapHealerImpl<PixelFormat::Gray8>>( nullptr, controls );
        case PixelFormat::Gray16:
            return std::make_shared<BitmapHealerImpl<PixelFormat::Gray16>>( nullptr, controls );

        default:
            throw std::invalid_argument( "BitmapHealer: Unsupported pixel format" );
    }
}

IBitmapPtr BitmapHealer::ApplyTransform( IBitmapPtr pSrcBitmap, const Settings& controls )
{
    auto pBitmapHealer = Create( pSrcBitmap, controls );
    return pBitmapHealer->RunAndGetBitmap();
}

ACMB_NAMESPACE_END