#include "HaloRemovalTransform.h"

BaseHaloRemovalTransform::BaseHaloRemovalTransform( IBitmapPtr pSrcBitmap, float intensity )
: BaseTransform(pSrcBitmap)
, _intensity (std::clamp( intensity, 0.0f, 1.0f ))
{
    if ( GetColorSpace( pSrcBitmap->GetPixelFormat() ) != ColorSpace::RGB )
        throw std::invalid_argument( "unsupported pixel format" );
}

std::shared_ptr<BaseHaloRemovalTransform> BaseHaloRemovalTransform::Create( IBitmapPtr pSrcBitmap, float intensity )
{
    switch ( pSrcBitmap->GetPixelFormat() )
    {
        case PixelFormat::RGB24:
            return std::make_shared<HaloRemovalTransform<PixelFormat::RGB24>>( std::static_pointer_cast<Bitmap<PixelFormat::RGB24>>(pSrcBitmap), intensity );
        case PixelFormat::RGB48:
            return std::make_shared<HaloRemovalTransform<PixelFormat::RGB48>>( std::static_pointer_cast< Bitmap<PixelFormat::RGB48> >( pSrcBitmap ), intensity );
        default:
            throw std::invalid_argument( "unsupported pixel format" );
    }
}

IBitmapPtr BaseHaloRemovalTransform::RemoveHalo( IBitmapPtr pSrcBitmap, float intensity )
{
    auto pRemoval = Create( pSrcBitmap, intensity );
    return pRemoval->RunAndGetBitmap();
}
