#pragma once
#include "basetransform.h"

ACMB_NAMESPACE_BEGIN

class SaturationTransform : public BaseTransform
{
public:
    using Settings = float;

protected:
    float _intensity = 1.0f;
    SaturationTransform( IBitmapPtr pSrcBitmap, float intensity );

public:
    static std::shared_ptr<SaturationTransform> Create( IBitmapPtr pSrcBitmap, float intensity );
    static std::shared_ptr<SaturationTransform> Create( PixelFormat pixelFormat, float intensity );
    static IBitmapPtr Saturate( IBitmapPtr pSrcBitmap, float intensity );
};

ACMB_NAMESPACE_END