#include "HaloRemovalTransform.h"
#include "HistogramBuilder.h"
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

ACMB_NAMESPACE_BEGIN

template <PixelFormat pixelFormat>
class HaloRemovalTransform_ final : public HaloRemovalTransform
{
    using ChannelType = typename PixelFormatTraits<pixelFormat>::ChannelType;

public:

    HaloRemovalTransform_( std::shared_ptr <Bitmap<pixelFormat>> pSrcBitmap, float intensity, float bgL, float peakHue, float sigma )
    : HaloRemovalTransform( pSrcBitmap, intensity, bgL, peakHue, sigma )
    {
    }
    virtual void Run() override
    {

        auto pSrcBitmap = std::static_pointer_cast< Bitmap<pixelFormat> >( _pSrcBitmap );

        const float a = ( 4 * _intensity ) / ( 4 * _bgL - ( 1 + _bgL ) * ( 1 + _bgL ) );
        const float b = -a * ( 1 + _bgL );
        const float c = a * _bgL;

        oneapi::tbb::parallel_for( oneapi::tbb::blocked_range<int>( 0, _pSrcBitmap->GetHeight() ), [pSrcBitmap, a, b, c, this] ( const oneapi::tbb::blocked_range<int>& range )
        {
            for ( int i = range.begin(); i < range.end(); ++i )
            {
                for ( size_t j = 0; j < pSrcBitmap->GetWidth(); ++j )
                {
                    auto rgb = std::span<ChannelType, 3>( pSrcBitmap->GetScanline( i ) + j * 3, 3 );
                    auto hsl = RgbToHsl( rgb );

                    const auto lbound = _peakHue - 3 * _sigma;
                    const auto ubound = _peakHue + 3 * _sigma;

                    float hue = hsl[0];

                    if ( lbound < 0 && hsl[0] > lbound + 360 )
                    {
                        hue -= 360;

                    }
                    else if ( ubound > 360 && hsl[0] < ubound - 360 )
                    {
                        hue += 360;
                    }

                    if ( hue > lbound && hue < ubound )
                    {

                        //hsl[2] *= ( 1 - _intensity * normalDist( hue, _peakHue, 1, _sigma ) );
                        auto coef = a * hsl[2] * hsl[2] + b * hsl[2] + c;
                        if ( hsl[2] > _bgL )
                        {
                            //auto coef = ( hsl[2] - bgL ) / ( 1 - bgL ) * 0.5f;


                            //auto coef = normalDist( hsl[2], ( 1 + bgL ) / 2, _intensity, ( 1 - bgL ) / 6 );
                            hsl[2] = std::max( _bgL, hsl[2] * ( 1 - _intensity * hsl[1] * coef * normalDist( hue, _peakHue, 1, _sigma ) ) );
                        }
                        /*else
                        {
                        hsl[2] = std::min( _bgL, hsl[2] * ( 1 + _intensity * hsl[1] * coef * normalDist( hue, _peakHue, 1, _sigma ) ) );
                        }*/

                        hsl[1] *= ( 1 - _intensity * normalDist( hue, _peakHue, 1, _sigma ) );
                        HslToRgb( hsl, rgb );
                    }
                }
            }
        } );
        _pDstBitmap = _pSrcBitmap;
    }
};

HaloRemovalTransform::HaloRemovalTransform( IBitmapPtr pSrcBitmap, float intensity, float bgL, float peakHue, float sigma )
: BaseTransform(pSrcBitmap)
, _intensity (std::clamp( intensity, 0.0f, 1.0f ))
, _peakHue(std::clamp(peakHue, 0.0f, 360.0f))
, _sigma(std::clamp(sigma, 0.0f, 60.0f))
, _bgL(std::clamp(bgL, 0.0f, 1.0f))
{
    if ( GetColorSpace( pSrcBitmap->GetPixelFormat() ) != ColorSpace::RGB )
        throw std::invalid_argument( "unsupported pixel format" );
}

std::shared_ptr<HaloRemovalTransform> HaloRemovalTransform::Create( IBitmapPtr pSrcBitmap, float intensity, float bgL, float peakHue, float sigma )
{
    if ( !pSrcBitmap )
        throw std::invalid_argument( "pSrcBitmap is null" );

    switch ( pSrcBitmap->GetPixelFormat() )
    {
        case PixelFormat::RGB24:
            return std::make_shared<HaloRemovalTransform_<PixelFormat::RGB24>>( std::static_pointer_cast<Bitmap<PixelFormat::RGB24>>(pSrcBitmap), intensity, bgL, peakHue, sigma );
        case PixelFormat::RGB48:
            return std::make_shared<HaloRemovalTransform_<PixelFormat::RGB48>>( std::static_pointer_cast< Bitmap<PixelFormat::RGB48> >( pSrcBitmap ), intensity, bgL, peakHue, sigma );
        default:
            throw std::invalid_argument( "unsupported pixel format" );
    }
}

IBitmapPtr HaloRemovalTransform::RemoveHalo( IBitmapPtr pSrcBitmap, float intensity, float bgL, float peakHue, float sigma )
{
    auto pRemoval = Create( pSrcBitmap, intensity, bgL, peakHue, sigma );
    return pRemoval->RunAndGetBitmap();
}

IBitmapPtr HaloRemovalTransform::AutoRemove( IBitmapPtr pSrcBitmap, float intensity )
{
    auto pHistBuilder = HistorgamBuilder::Create( pSrcBitmap );
    pHistBuilder->BuildHistogram();
    std::array<uint16_t, 3> medianRgb =
    {
        uint16_t( pHistBuilder->GetChannelStatistics( 0 ).centils[50] ),
        uint16_t( pHistBuilder->GetChannelStatistics( 1 ).centils[50] ),
        uint16_t( pHistBuilder->GetChannelStatistics( 2 ).centils[50] )
    };
    auto medianHsl = RgbToHsl<uint16_t>( std::span( medianRgb ) );
    auto pRes = HaloRemovalTransform::RemoveHalo( pSrcBitmap, intensity, medianHsl[2] * 2, 250, 10 );
    pRes = HaloRemovalTransform::RemoveHalo( pRes, intensity, medianHsl[2] * 2, 270, 20 );
    return HaloRemovalTransform::RemoveHalo( pRes, intensity * 1.2f, medianHsl[2] * 2, 300, 10 );
}

ACMB_NAMESPACE_END
