#pragma once

#include "FastAligner.h"
#include "../Core/bitmap.h"
#include "../Geometry/triangle.h"
#include "../AGG/agg_trans_affine.h"
#include "../Tests/test.h"

#include <array>

ACMB_TESTS_NAMESPACE_BEGIN
class TestStacker;
ACMB_TESTS_NAMESPACE_END

ACMB_NAMESPACE_BEGIN

class ImageDecoder;

struct StackingDatum
{
    std::shared_ptr<ImageDecoder> pDecoder;
    std::vector<std::vector<Star>> stars;
    uint64_t totalStarCount;
};

class Stacker
{
    template <PixelFormat pixelFormat> friend class AddingBitmapHelper;
    template <PixelFormat pixelFormat> friend class AddingBitmapWithAlignmentHelper;
    template <PixelFormat pixelFormat> friend class GeneratingResultHelper;
    friend class AlignmentHelper;

    using TriangleTransformPair = std::pair<Triangle, agg::trans_affine>;
    using GridCell = std::vector<TriangleTransformPair>;
    using Grid = std::vector<GridCell>;

    Grid _grid;
    std::vector<StackingDatum> _stackingData;

    std::vector<float> _means;
    std::vector<float> _devs;
    std::vector<uint16_t> _counts;

    uint32_t _width = 0;
    uint32_t _height = 0;

    static const size_t gridSize = 100;
    uint32_t _gridWidth = 0;
    uint32_t _gridHeight = 0;

    double _alignmentError = 2.0;

    bool _enableDeaberration;

    std::vector<std::shared_ptr<FastAligner>> _aligners;
    MatchMap _matches;

    IBitmapPtr _pDarkFrame;


    void ChooseTriangle(PointF p, std::pair<Triangle, agg::trans_affine>& lastPair, const GridCell& trianglePairs);
    void StackWithAlignment(IBitmapPtr pTargetBitmap, uint32_t i);

public:

    Stacker(std::vector<std::shared_ptr<ImageDecoder>> decoders, bool enableDeaberration = false);

    void Registrate(double threshold = 40, uint32_t minStarSize = 5, uint32_t maxStarSize = 25);
    std::shared_ptr<IBitmap>  RegistrateAndStack(double threshold = 40, uint32_t minStarSize = 5, uint32_t maxStarSize = 25);
    std::shared_ptr<IBitmap> Stack(bool doAlignment);   

    void SetDarkFrame( IBitmapPtr pDarkFrame );
    TEST_ACCESS(Stacker);

};

ACMB_NAMESPACE_END
