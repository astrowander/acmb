#pragma once

#include "basetransform.h"
#include "../AGG/agg_trans_affine.h"

ACMB_NAMESPACE_BEGIN

class AffineTransform : public BaseTransform
{
public:
    struct Settings
    {
        agg::trans_affine transform;
        IColorPtr pBgColor;
    };
protected:
    Settings _settings;

    AffineTransform( IBitmapPtr pSrcBitmap, const Settings& settings );

public:
    static std::shared_ptr<AffineTransform> Create( IBitmapPtr pSrcBitmap, const Settings& settings );
    static std::shared_ptr<AffineTransform> Create( PixelFormat, const Settings& settings );
    static IBitmapPtr ApplyTransform( IBitmapPtr pSrcBitmap, const Settings& settings );
};

ACMB_NAMESPACE_END