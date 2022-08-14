#include "HaloRemovalTransform.h"
#include "HistogramBuilder.h"

BaseHaloRemovalTransform::BaseHaloRemovalTransform( IBitmapPtr pSrcBitmap, float intensity, float bgL, float peakHue, float sigma )
: BaseTransform(pSrcBitmap)
, _intensity (std::clamp( intensity, 0.0f, 1.0f ))
, _bgL(std::clamp(bgL, 0.0f, 1.0f))
, _peakHue(std::clamp(peakHue, 0.0f, 360.0f))
, _sigma(std::clamp(sigma, 0.0f, 60.0f))
{
    if ( GetColorSpace( pSrcBitmap->GetPixelFormat() ) != ColorSpace::RGB )
        throw std::invalid_argument( "unsupported pixel format" );
}

std::shared_ptr<BaseHaloRemovalTransform> BaseHaloRemovalTransform::Create( IBitmapPtr pSrcBitmap, float intensity, float bgL, float peakHue, float sigma )
{
    switch ( pSrcBitmap->GetPixelFormat() )
    {
        case PixelFormat::RGB24:
            return std::make_shared<HaloRemovalTransform<PixelFormat::RGB24>>( std::static_pointer_cast<Bitmap<PixelFormat::RGB24>>(pSrcBitmap), intensity, bgL, peakHue, sigma );
        case PixelFormat::RGB48:
            return std::make_shared<HaloRemovalTransform<PixelFormat::RGB48>>( std::static_pointer_cast< Bitmap<PixelFormat::RGB48> >( pSrcBitmap ), intensity, bgL, peakHue, sigma );
        default:
            throw std::invalid_argument( "unsupported pixel format" );
    }
}

IBitmapPtr BaseHaloRemovalTransform::RemoveHalo( IBitmapPtr pSrcBitmap, float intensity, float bgL, float peakHue, float sigma )
{
    auto pRemoval = Create( pSrcBitmap, intensity, bgL, peakHue, sigma );
    return pRemoval->RunAndGetBitmap();
}

IBitmapPtr BaseHaloRemovalTransform::AutoRemove( IBitmapPtr pSrcBitmap, float intensity )
{
    auto pHistBuilder = BaseHistorgamBuilder::Create( pSrcBitmap );
    pHistBuilder->BuildHistogram();
    std::array<uint16_t, 3> medianRgb =
    {
        uint16_t( pHistBuilder->GetChannelStatistics( 0 ).centils[50] ),
        uint16_t( pHistBuilder->GetChannelStatistics( 1 ).centils[50] ),
        uint16_t( pHistBuilder->GetChannelStatistics( 2 ).centils[50] )
    };
    auto medianHsl = RgbToHsl<uint16_t>( std::span( medianRgb ) );
    auto pRes = BaseHaloRemovalTransform::RemoveHalo( pSrcBitmap, intensity, medianHsl[2] * 2, 250, 10 );
    pRes = BaseHaloRemovalTransform::RemoveHalo( pRes, intensity, medianHsl[2] * 2, 270, 20 );
    return BaseHaloRemovalTransform::RemoveHalo( pRes, intensity * 1.2f, medianHsl[2] * 2, 300, 10 );
}
