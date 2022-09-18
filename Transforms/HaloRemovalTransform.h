#pragma once

#include "basetransform.h"
#include "./../Tools/mathtools.h"

ACMB_NAMESPACE_BEGIN

class HaloRemovalTransform : public BaseTransform
{
protected:
    float _intensity;
    float _peakHue;
    float _sigma;
    float _bgL;

    HaloRemovalTransform( IBitmapPtr pSrcBitmap, float intensity, float bgL, float peakHue, float sigma );

public:    
    static std::shared_ptr<HaloRemovalTransform> Create( IBitmapPtr pSrcBitmap, float intensity, float bgL = 0.3f, float peakHue = 285.0f, float sigma = 40.0f );
    static IBitmapPtr RemoveHalo( IBitmapPtr pSrcBitmap, float intensity, float bgL = 0.3f, float peakHue = 285.0f, float sigma = 40.f );
    static IBitmapPtr AutoRemove( IBitmapPtr pSrcBitmap, float intensity );
};

ACMB_NAMESPACE_END