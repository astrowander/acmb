#pragma once
#include "basetransform.h"
#include "../Geometry/size.h"

ACMB_NAMESPACE_BEGIN

class CenterObjectTransform : public BaseTransform
{
public:
    struct Settings
    {
        Size dstSize;
        float threshold;
    };
protected:
    Settings settings_;
    CenterObjectTransform( IBitmapPtr pSrcBitmap, const Settings& settings );
public:
    static std::shared_ptr<CenterObjectTransform> Create( IBitmapPtr pSrcBitmap, const Settings& settings );
    static std::shared_ptr<CenterObjectTransform> Create( PixelFormat srcPixelFormat, const Settings& settings );
    static IBitmapPtr CenterObject( IBitmapPtr pSrcBitmap, const Settings& settings );
    virtual void CalcParams( std::shared_ptr<ImageParams> pParams ) override;
};

ACMB_NAMESPACE_END
