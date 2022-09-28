#pragma once

#include "FastAligner.h"
#include "../Core/bitmap.h"
#include "../Core/pipeline.h"
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
    Pipeline pipeline;
    std::vector<std::vector<Star>> stars;
    uint64_t totalStarCount;
};

class Stacker : public IPipelineElement
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

    static const size_t gridSize = 100;
    uint32_t _gridWidth = 0;
    uint32_t _gridHeight = 0;

    double _alignmentError = 2.0;

    bool _doAlignment = true;

    std::vector<std::shared_ptr<FastAligner>> _aligners;
    MatchMap _matches;

    IBitmapPtr _pDarkFrame;

    double _threshold = 25.0;
    uint32_t _minStarSize = 5;
    uint32_t _maxStarSize = 25;

    void ChooseTriangle(PointF p, std::pair<Triangle, agg::trans_affine>& lastPair, const GridCell& trianglePairs);
    void StackWithAlignment(IBitmapPtr pTargetBitmap, uint32_t i);

public:

    Stacker(const std::vector<Pipeline>& pipeline);

    void Registrate();
    std::shared_ptr<IBitmap>  RegistrateAndStack();
    std::shared_ptr<IBitmap> Stack();   

    void SetDarkFrame( IBitmapPtr pDarkFrame );

    double GetThreshold() { return _threshold; }
    void SetThreshold( double threshold ) { _threshold = threshold; };
    double GetMinStarSize() { return _minStarSize; }
    void SetMinStarSize( uint32_t minStarSize ) {  _minStarSize = minStarSize; };
    double GetMaxStarSize() { return _threshold; }
    void SetMaxStarSize( uint32_t maxStarSize ) { _maxStarSize = maxStarSize; };

    bool GetDoAlignment() { return _doAlignment; }
    void SetDoAlignment( bool val ) { _doAlignment = val; }

    virtual IBitmapPtr ProcessBitmap( IBitmapPtr pSrcBitmap = nullptr ) override;
        
    TEST_ACCESS(Stacker);
};

ACMB_NAMESPACE_END
