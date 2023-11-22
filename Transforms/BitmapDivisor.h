#pragma once
#include "basetransform.h"

ACMB_NAMESPACE_BEGIN
/// <summary>
/// Divides the source image by the other one
/// </summary>
class BitmapDivisor : public BaseTransform
{
public:
    struct Settings
    {
        IBitmapPtr pDivisor;
        float intensity = 100.0f;
    };

protected:
    Settings _settings;
    BitmapDivisor( IBitmapPtr pSrcBitmap, const Settings& settings );

public:
    /// Creates an instance with source bitmap and bitmap that will be a divisor
    static std::shared_ptr<BitmapDivisor> Create( IBitmapPtr pSrcBitmap, const Settings& settings );
    /// Creates an instance with pixel format of source bitmap (need to set it later) and bitmap that will a divisor
    static std::shared_ptr<BitmapDivisor> Create( PixelFormat srcPixelFormat, const Settings& settings );
    /// Divides the source image by the other one
    static IBitmapPtr Divide( IBitmapPtr pSrcBitmap, const Settings& settings );
};

ACMB_NAMESPACE_END