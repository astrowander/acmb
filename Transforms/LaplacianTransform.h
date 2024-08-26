#pragma once

#include "basetransform.h"

ACMB_NAMESPACE_BEGIN

class ConvolutionTransform : public BaseTransform
{
public:
    struct Settings
    {
        Size kernelSize;
        std::vector<float> kernel;
    };
protected:
    Settings _settings;
    ConvolutionTransform( IBitmapPtr pSrcBitmap, const Settings& settings );

public:
    static std::shared_ptr<ConvolutionTransform> Create( IBitmapPtr pSrcBitmap, const Settings& settings );
    static std::shared_ptr<ConvolutionTransform> Create( PixelFormat pixelFormat, const Settings& settings );
    static IBitmapPtr ApplyConvolution( IBitmapPtr pSrcBitmap, const Settings& settings );

    static IBitmapPtr ApplyLaplacian( IBitmapPtr pSrcBitmap );

    virtual void CalcParams( std::shared_ptr<ImageParams> pParams ) override;
};

ACMB_NAMESPACE_END
