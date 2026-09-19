#pragma once

#include "basetransform.h"

ACMB_NAMESPACE_BEGIN

class DeconvTransform : public BaseTransform
{
public:
    struct Settings
    {
        double _thresholdPercents = 40;
        int _minStarSize = 5;
        int _maxStarSize = 25;
        int _iterations = 10;
        double _intensity = 1.0;
    };

protected:
    Settings _settings;

    DeconvTransform(IBitmapPtr pSrcBitmap, const Settings& settings);

public:
    static std::shared_ptr<DeconvTransform> Create(IBitmapPtr pSrcBitmap, const Settings& settings);
    static std::shared_ptr<DeconvTransform> Create(PixelFormat, const Settings& settings);
    static IBitmapPtr ApplyTransform(IBitmapPtr pSrcBitmap, const Settings& settings);

    static Settings Interpolate(const Settings& a, const Settings& b, double t);
};

ACMB_NAMESPACE_END