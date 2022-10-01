#pragma once

#include "basetransform.h"
#include "./../Tools/mathtools.h"

ACMB_NAMESPACE_BEGIN

/// <summary>
/// Removes colored halos around the bright stars
/// </summary>
class HaloRemovalTransform : public BaseTransform
{
protected:
    float _intensity;
    float _peakHue;
    float _sigma;
    float _bgL;

    HaloRemovalTransform( IBitmapPtr pSrcBitmap, float intensity, float bgL, float peakHue, float sigma );

public:
    /// <summary>
    /// Creates instance with given parameters
    /// </summary>
    /// <param name="pSrcBitmap">Source bitmap </param>
    /// <param name="intensity">intensity of transformation. Changes from 0 to 1</param>
    /// <param name="bgL">minimal luminosity. Below this value transform is not applied. Changes from 0 to 1</param>
    /// <param name="peakHue"> hue in the HSL color space, where effect is maximal</param>
    /// <param name="sigma">standard deviation. The more the wider range of hues are influenced</param>
    static std::shared_ptr<HaloRemovalTransform> Create( IBitmapPtr pSrcBitmap, float intensity, float bgL = 0.3f, float peakHue = 285.0f, float sigma = 40.0f );

    /// <summary>
    /// Runs transform and return result
    /// </summary>
    /// <param name="pSrcBitmap">Source bitmap </param>
    /// <param name="intensity">intensity of transformation. Changes from 0 to 1</param>
    /// <param name="bgL">minimal luminosity. Below this value transform is not applied. Changes from 0 to 1</param>
    /// <param name="peakHue"> hue in the HSL color space, where effect is maximal</param>
    /// <param name="sigma">standard deviation. The more the wider range of hues are influenced</param>
    static IBitmapPtr RemoveHalo( IBitmapPtr pSrcBitmap, float intensity, float bgL = 0.3f, float peakHue = 285.0f, float sigma = 40.f );

    /// Runs transform with default parameters and returns result
    static IBitmapPtr AutoRemove( IBitmapPtr pSrcBitmap, float intensity );
};

/// <summary>
/// This class is needed for compatibility with pipelines. It applies default transformations
/// </summary>
class AutoHaloRemoval : public BaseTransform
{
public:
    using Settings = float;

private:
    float _intensity;

public:
    /// <summary>
    /// Runs transform and return result
    /// </summary>
    /// <param name="pSrcBitmap">Source bitmap </param>
    /// <param name="intensity">intensity of transformation. Changes from 0 to 1</param>
    AutoHaloRemoval( IBitmapPtr pSrcBitmap, float intensity );
    /// Runs the transformation
    virtual void Run() override;
    /// Creates instance with given pixel format. Source bitmap must be set later
    static std::shared_ptr<AutoHaloRemoval> Create( PixelFormat pixelFormat, float intensity );
};

ACMB_NAMESPACE_END