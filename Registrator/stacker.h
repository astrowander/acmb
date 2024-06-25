#pragma once

#include "BaseStacker.h"
#include "../Core/bitmap.h"
#include "../Tests/test.h"

ACMB_TESTS_NAMESPACE_BEGIN
class TestStacker;
ACMB_TESTS_NAMESPACE_END

ACMB_NAMESPACE_BEGIN
class Registrator;

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

public:
    /// creates an instance with the given images
    Stacker(const std::vector<Pipeline>& pipelines, StackMode stackMode);

    Stacker( const ImageParams& imageParams, StackMode stackMode );

    virtual void CallAddBitmapHelper( IBitmapPtr pBitmap ) override;
    virtual void CallAddBitmapWithAlignmentHelper( IBitmapPtr pBitmap ) override;
    virtual IBitmapPtr CallGeneratingResultHelper() override;

    IBitmapPtr GenerateDeviationMap() const;
        
    TEST_ACCESS(Stacker);
};

ACMB_NAMESPACE_END
