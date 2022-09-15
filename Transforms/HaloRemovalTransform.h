#pragma once

#include "basetransform.h"
#include "./../Tools/mathtools.h"

class BaseHaloRemovalTransform : public BaseTransform
{
protected:
    float _intensity;
    float _peakHue;
    float _sigma;
    float _bgL;

    BaseHaloRemovalTransform( IBitmapPtr pSrcBitmap, float intensity, float bgL, float peakHue, float sigma );

public:    
    static std::shared_ptr<BaseHaloRemovalTransform> Create( IBitmapPtr pSrcBitmap, float intensity, float bgL = 0.3f, float peakHue = 285.0f, float sigma = 40.0f );
    static IBitmapPtr RemoveHalo( IBitmapPtr pSrcBitmap, float intensity, float bgL = 0.3f, float peakHue = 285.0f, float sigma = 40.f );
    static IBitmapPtr AutoRemove( IBitmapPtr pSrcBitmap, float intensity );
};

template <PixelFormat pixelFormat>
class HaloRemovalTransform final: public BaseHaloRemovalTransform
{
    using ChannelType = typename PixelFormatTraits<pixelFormat>::ChannelType;   

public:
    
    HaloRemovalTransform( std::shared_ptr <Bitmap<pixelFormat>> pSrcBitmap, float intensity, float bgL, float peakHue, float sigma );
    void Run() override;
};
