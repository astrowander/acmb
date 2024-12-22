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
        _pDstBitmap = pSrcBitmap->Clone();
    }

    void ApplyPatch( const Patch& patch )
    {

        auto pSrcBitmap = std::static_pointer_cast< Bitmap<pixelFormat> >( _pSrcBitmap );        
        auto pDstBitmap = std::static_pointer_cast< Bitmap<pixelFormat> >(_pDstBitmap);

        parallel_for( blocked_range<int>( -patch.radius, patch.radius + 1 ), [&] ( const blocked_range<int>& r )
        {
            for ( int dy = r.begin(); dy < r.end(); ++dy )
            {
                for ( int dx = -patch.radius; dx <= patch.radius; ++dx )
                {
                    Point srcPos = patch.from + Point( dx, dy );
                    if ( srcPos.x < 0 || srcPos.y < 0 || srcPos.x >= int( pSrcBitmap->GetWidth() ) || srcPos.y >= int( pSrcBitmap->GetHeight() ) )
                        continue;

                    Point dstPos = patch.to + Point( dx, dy );

                    if ( dstPos.x < 0 || dstPos.y < 0 || dstPos.x >= int( pDstBitmap->GetWidth() ) || dstPos.y >= int( pDstBitmap->GetHeight() ) )
                        continue;

                    const float dist = sqrt( dx * dx + dy * dy );

                    if ( dist >= patch.radius )
                        continue;

                    float coeff = 1.0f;

                    if ( dist > patch.radius * 0.5f )
                        coeff = pow( 2.0f * (patch.radius - dist) / patch.radius, patch.gamma );

                    const ChannelType* pSrcScanline = pSrcBitmap->GetScanline( srcPos.y );
                    ChannelType* pDstScanline = pDstBitmap->GetScanline( dstPos.y );

                    for ( int i = 0; i < channelCount; ++i )
                    {
                        ChannelType dstValue = pDstScanline[dstPos.x * channelCount + i];
                        ChannelType srcValue = pSrcScanline[srcPos.x * channelCount + i];

                        pDstScanline[dstPos.x * channelCount + i] =
                            ChannelType( std::clamp<float>( srcValue * coeff +
                                                            dstValue * (1 - coeff) + 0.5f, 0, channelMax ) );
                    }

                }
            }
        } );
    }

    virtual void Run() override
    {
        for ( const auto& patch : _patches )
            ApplyPatch( patch );
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