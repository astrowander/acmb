#pragma once
#include "basetransform.h"

ACMB_NAMESPACE_BEGIN

class WarpTransform : public BaseTransform
{
public:
    struct Settings
    {
        std::array<PointF, 16> controls =
        { PointF{ 0.0, 0.0 }, { 1.0 / 3.0, 0.0 },       { 2.0 / 3.0, 0.0 },       { 1.0, 0.0 },
          { 0.0, 1.0 / 3.0 }, { 1.0 / 3.0, 1.0 / 3.0 }, { 2.0 / 3.0, 1.0 / 3.0 }, { 1.0, 1.0 / 3.0 },
          { 0.0, 2.0 / 3.0 }, { 1.0 / 3.0, 2.0 / 3.0 }, { 2.0 / 3.0, 2.0 / 3.0 }, { 1.0, 2.0 / 3.0 },
          { 0.0, 1.0 },       { 1.0 / 3.0, 1.0 },       { 2.0 / 3.0, 1.0 },       { 1.0, 1.0 } };
        IColorPtr pBgColor;
    };
protected:
    Settings _settings;

    WarpTransform( IBitmapPtr pSrcBitmap, const Settings& controls );

public:
    static std::shared_ptr<WarpTransform> Create( IBitmapPtr pSrcBitmap, const Settings& controls );
    static std::shared_ptr<WarpTransform> Create( PixelFormat, const Settings& controls );
    static IBitmapPtr Warp( IBitmapPtr pSrcBitmap, const Settings& controls );
};


ACMB_NAMESPACE_END