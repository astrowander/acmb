#pragma once

#include "basetransform.h"
#include "./../Core/IParallel.h"

class BaseHaloRemovalTransform : public BaseTransform
{
protected:
    float _intensity;

public:
    BaseHaloRemovalTransform( IBitmapPtr pSrcBitmap, float intensity );
    static std::shared_ptr<BaseHaloRemovalTransform> Create( IBitmapPtr pSrcBitmap, float intensity );
    static IBitmapPtr RemoveHalo( IBitmapPtr pSrcBitmap, float intensity );
};

template <PixelFormat pixelFormat>
class HaloRemovalTransform : public BaseHaloRemovalTransform, public IParallel
{
private:
    HaloRemovalTransform( std::shared_ptr <Bitmap<pixelFormat>> pSrcBitmap, float intensity)
    : BaseHaloRemovalTransform(pSrcBitmap, intensity)
    { }

    void Job( uint32_t i ) override
    {
        auto pSrcBitmap = std::static_pointer_cast< Bitmap<pixelFormat> >( _pSrcBitmap );
        for ( size_t j = 0; j < pSrcBitmap->GetWidth(); ++j )
        {

        }
    }
};
