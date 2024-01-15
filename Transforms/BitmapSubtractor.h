#pragma once
#include "basetransform.h"

ACMB_NAMESPACE_BEGIN
/// <summary>
/// Subtracts given image from the source
/// </summary>
class BitmapSubtractor : public BaseTransform
{
public:
    struct Settings
    {
        IBitmapPtr pBitmapToSubtract;
        float intensity = 100.0f;
    };
protected:
    Settings _settings;
    BitmapSubtractor( IBitmapPtr pSrcBitmap, const Settings& settings );

public:
    /// Creates an instance with source bitmap and bitmap that will be subtracted
    static std::shared_ptr<BitmapSubtractor> Create( IBitmapPtr pSrcBitmap, const Settings& settings );
    /// Creates an instance with pixel format of source bitmap (need to set it later) and bitmap that will be subtracted
    static std::shared_ptr<BitmapSubtractor> Create( PixelFormat srcPixelFormat, const Settings& settings );
    /// Subtracts given bitmap from the source and returns result
    static IBitmapPtr Subtract( IBitmapPtr pSrcBitmap, const Settings& settings );
};

ACMB_NAMESPACE_END