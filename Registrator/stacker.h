#pragma once

#include "BaseStacker.h"
#include "../Core/bitmap.h"
#include "../Tests/test.h"

ACMB_TESTS_NAMESPACE_BEGIN
class TestStacker;
ACMB_TESTS_NAMESPACE_END

ACMB_NAMESPACE_BEGIN

class ImageDecoder;

/// <summary>
/// Combines a bunch of images to the one stacked image.
/// </summary>
class Stacker : public BaseStacker
{
    template <PixelFormat pixelFormat> friend class AddingBitmapHelper;
    template <PixelFormat pixelFormat> friend class AddingBitmapWithAlignmentHelper;
    template <PixelFormat pixelFormat> friend class GeneratingResultHelper;

    std::vector<float> _means;
    std::vector<float> _devs;
    std::vector<uint16_t> _counts;

    void ChooseTriangle(PointF p, std::pair<Triangle, agg::trans_affine>& lastPair, const GridCell& trianglePairs);
    void StackWithAlignment(IBitmapPtr pTargetBitmap, uint32_t i);

public:
    /// creates an instance with the given images
    Stacker(const std::vector<Pipeline>& pipelines, StackMode stackMode);

    /// detects stars and stacks images in one time
    virtual std::shared_ptr<IBitmap>  RegistrateAndStack() override;
    /// stacks registered images
    virtual std::shared_ptr<IBitmap> Stack() override;

    /// needed for compatibility with pipeline API
    virtual IBitmapPtr ProcessBitmap( IBitmapPtr pSrcBitmap = nullptr ) override;
        
    TEST_ACCESS(Stacker);
};

ACMB_NAMESPACE_END
