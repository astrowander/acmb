#include "WarpTransform.h"
#include "../Core/bitmap.h"
#include "../Tools/mathtools.h"
#include <algorithm>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

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

        std::vector<PointF> topCurve( width );
        std::vector<PointF> leftCurve( height );
        std::vector<PointF> rightCurve( height );
        std::vector<PointF> bottomCurve( width );

        auto pTopPixel = pDstBitmap->GetScanline( 0 );
        auto pBottomPixel = pDstBitmap->GetScanline( height - 1 );

        for ( uint32_t x = 0; x < width; ++x )
        {
            topCurve[x] = CubicBezier( _settings.controls[0], _settings.controls[1], _settings.controls[2], _settings.controls[3], double( x ) / ( width - 1 ) );            
            bottomCurve[x] = CubicBezier( _settings.controls[8], _settings.controls[9], _settings.controls[10], _settings.controls[11], double( x ) / ( width - 1 ) );

            for ( uint32_t ch = 0; ch < channelCount; ++ch )
            {
                *pTopPixel++ = pSrcBitmap->GetInterpolatedChannel( topCurve[x].x * ( width - 1 ), topCurve[x].y * ( height - 1 ), ch);
                *pBottomPixel++= pSrcBitmap->GetInterpolatedChannel( bottomCurve[x].x * ( width - 1 ), bottomCurve[x].y * ( height - 1 ), ch ); 
            }
        }

        auto pLeftPixel = pDstBitmap->GetScanline( 0 );
        auto pRightPixel = pDstBitmap->GetScanline( 0 ) + ( width - 1 ) * channelCount;

        for ( uint32_t y = 0; y < height; ++y )
        {
            leftCurve[y] = CubicBezier( _settings.controls[0], _settings.controls[4], _settings.controls[6], _settings.controls[8], double( y ) / ( height - 1 ) );
            rightCurve[y] = CubicBezier( _settings.controls[3], _settings.controls[5], _settings.controls[7], _settings.controls[11], double( y ) / ( height - 1 ) );            

            for ( uint32_t ch = 0; ch < channelCount; ++ch )
            {
                *pLeftPixel++ = pSrcBitmap->GetInterpolatedChannel( leftCurve[y].x * ( width - 1 ), leftCurve[y].y * ( height - 1 ), ch );
                *pRightPixel++ = pSrcBitmap->GetInterpolatedChannel( rightCurve[y].x * ( width - 1 ), rightCurve[y].y * ( height - 1 ), ch );
            }

            pLeftPixel += channelCount * ( width - 1 );
            pRightPixel += channelCount * ( width - 1 );
        }

        for (uint32_t y = 1; y < height - 1; ++y)
        {
            auto pPixel = pDstBitmap->GetScanline( y );
            for (uint32_t x = 1; x < width - 1; ++x)
            {
                if ( x == 320 )
                    x = x;

                PointF P{ double( x ) / ( width - 1 ) , double( y ) / ( height - 1 ) };

                struct Anchor
                {
                    PointF dr;
                    double affinity = 0.0;
                };

                Anchor anchors[4];
                anchors[0].dr = topCurve[x] - PointF{ P.x, 0 };
                anchors[0].affinity = P.y;

                anchors[1].dr = rightCurve[y] - PointF{ 1, P.y };
                anchors[1].affinity = 1 - P.x;

                anchors[2].dr = bottomCurve[x] - PointF{ P.x, 1 };
                anchors[2].affinity = 1 - P.y;

                anchors[3].dr = leftCurve[y] - PointF{ 0, P.y };
                anchors[3].affinity = P.x;

                PointF P1 = P;
                for ( uint32_t i = 0; i < 4; ++i )
                {
                    const double length = anchors[i].dr.Length();
                    if ( length > 0 )
                        P1 += (1 - 1 / anchors[i].affinity) * anchors[i].dr / length;
                }
                for ( uint32_t ch = 0; ch < channelCount; ++ch )
                {
                    pPixel[ch] = pSrcBitmap->GetInterpolatedChannel( P1.x * ( width - 1 ), P1.y * ( height - 1 ), ch );
                }

                pPixel += channelCount;
            }
        }

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