#include "HaloRemovalTransform.h"

BaseHaloRemovalTransform::BaseHaloRemovalTransform( IBitmapPtr pSrcBitmap, float intensity, float bgL, float peakHue, float sigma )
: BaseTransform(pSrcBitmap)
, _intensity (std::clamp( intensity, 0.0f, 1.0f ))
, _bgL(std::clamp(bgL, 0.0f, 0.1f))
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
    auto pRemoval = Create( pSrcBitmap, intensity );
    return pRemoval->RunAndGetBitmap();
}
