#include "WarpTransform.h"
#include "../Core/bitmap.h"
#include "../Tools/mathtools.h"
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <functional>

using namespace oneapi::tbb;

ACMB_NAMESPACE_BEGIN

template<PixelFormat pixelFormat>
class WarpTransform_ : public WarpTransform
{
    using ChannelType = typename PixelFormatTraits<pixelFormat>::ChannelType;
    static constexpr auto channelCount = PixelFormatTraits<pixelFormat>::channelCount;

public:
    WarpTransform_(std::shared_ptr<Bitmap<pixelFormat>> pSrcBitmap, const Settings& settings)
    : WarpTransform(pSrcBitmap, settings)
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

        std::array<std::function<float(float)>, 4> bernstein =
        {
            []( float u ) { const float inv = 1 - u; return inv * inv * inv; },
            []( float u ) { const float inv = 1 - u; return 3 * u * inv * inv; },
            []( float u ) { const float inv = 1 - u; return 3 * u * u * inv; },
            []( float u ) { return u * u * u; }
        };

        std::array< std::vector<float>, 4 > uTable;
        std::array< std::vector<float>, 4 > vTable;

        for ( int i = 0; i < 4; ++i )
        {
            uTable[i].resize( width );
            vTable[i].resize( height );

            for ( uint32_t x = 0; x < width; ++x )
                uTable[i][x] = bernstein[i]( float( x ) / (width - 1) );

            for ( uint32_t y = 0; y < height; ++y )
                vTable[i][y] = bernstein[i]( float( y ) / (height - 1));
        }

        Settings settings;
        for ( int i = 0; i < 16; ++i )
            settings.controls[i] = 2.0f * settings.controls[i] - _settings.controls[i];

        parallel_for( blocked_range<uint32_t>( 0, height ), [&] ( const blocked_range<uint32_t>& r )
        {
            for ( uint32_t y = r.begin(); y < r.end(); ++y )
            {
                auto pDstPixel = pDstBitmap->GetScanline( y );
                for ( uint32_t x = 0; x < width; ++x )
                {
                    PointF p;
                    for ( int i = 0; i < 4; ++i )
                    for ( int j = 0; j < 4; ++j )
                    {
                        p += settings.controls[i * 4 + j] * uTable[j][x] * vTable[i][y];
                    }

                    if ( p.x < 0 || p.x > 1 || p.y < 0 || p.y > 1 )
                    {
                        for ( uint32_t ch = 0; ch < channelCount; ++ch )
                            pDstPixel[ch] = _settings.pBgColor->GetChannel( ch );
                    }
                    else
                    {

                        p.x *= width - 1;
                        p.y *= height - 1;

                        for ( uint32_t ch = 0; ch < channelCount; ++ch )
                            pDstPixel[ch] = pSrcBitmap->GetInterpolatedChannel( p.x, p.y, ch );
                    }

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

WarpTransform::WarpTransform( IBitmapPtr pSrcBitmap, const Settings& controls )
: BaseTransform( pSrcBitmap )
, _settings( controls )
{}

std::shared_ptr<WarpTransform> WarpTransform::Create( IBitmapPtr pSrcBitmap, const Settings& controls )
{
    switch ( pSrcBitmap->GetPixelFormat() )
    {
    case PixelFormat::RGB24:
        return std::make_shared<WarpTransform_<PixelFormat::RGB24>>( std::static_pointer_cast<Bitmap<PixelFormat::RGB24>>( pSrcBitmap ), controls );
    case PixelFormat::RGB48:
        return std::make_shared<WarpTransform_<PixelFormat::RGB48>>( std::static_pointer_cast< Bitmap<PixelFormat::RGB48> >(pSrcBitmap), controls );
    case PixelFormat::Gray8:
        return std::make_shared<WarpTransform_<PixelFormat::Gray8>>( std::static_pointer_cast< Bitmap<PixelFormat::Gray8> >(pSrcBitmap), controls );
    case PixelFormat::Gray16:
        return std::make_shared<WarpTransform_<PixelFormat::Gray16>>( std::static_pointer_cast< Bitmap<PixelFormat::Gray16> >(pSrcBitmap), controls );

    default:
        throw std::invalid_argument( "WarpTransform: Unsupported pixel format" );
    }
}

std::shared_ptr<WarpTransform> WarpTransform::Create( PixelFormat pixelFormat, const Settings& controls )
{
    switch ( pixelFormat )
    {
    case PixelFormat::RGB24:
        return std::make_shared<WarpTransform_<PixelFormat::RGB24>>( nullptr, controls );
    case PixelFormat::RGB48:
        return std::make_shared<WarpTransform_<PixelFormat::RGB48>>( nullptr, controls );
    case PixelFormat::Gray8:
        return std::make_shared<WarpTransform_<PixelFormat::Gray8>>( nullptr, controls );
    case PixelFormat::Gray16:
        return std::make_shared<WarpTransform_<PixelFormat::Gray16>>( nullptr, controls );

    default:
        throw std::invalid_argument( "WarpTransform: Unsupported pixel format" );
    }
}

IBitmapPtr WarpTransform::Warp( IBitmapPtr pSrcBitmap, const Settings& controls )
{
    auto pWarpTransform = Create( pSrcBitmap, controls );
    return pWarpTransform->RunAndGetBitmap();
}
ACMB_NAMESPACE_END