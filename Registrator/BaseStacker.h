#pragma once
#include "FastAligner.h"
#include "./../AGG/agg_trans_affine.h"
#include "../Core/pipeline.h"
#include "../Geometry/triangle.h"

ACMB_NAMESPACE_BEGIN

struct StackingDatum
{
    Pipeline pipeline;
    std::vector<std::vector<Star>> stars;
    uint64_t totalStarCount;
};

enum class StackMode
{
    Light,
    LightNoAlign,
    DarkOrFlat
};

class BaseStacker : public IPipelineFirstElement
{
    friend class AlignmentHelper;

public:
    using TriangleTransformPair = std::pair<Triangle, agg::trans_affine>;
    using GridCell = std::vector<TriangleTransformPair>;
    using Grid = std::vector<GridCell>;

protected:
    Grid _grid;
    std::vector<StackingDatum> _stackingData;

    static const size_t gridSize = 100;
    uint32_t _gridWidth = 0;
    uint32_t _gridHeight = 0;

    std::vector<std::shared_ptr<FastAligner>> _aligners;
    MatchMap _matches;

    double _threshold = 25.0;
    uint32_t _minStarSize = 5;
    uint32_t _maxStarSize = 25;

    StackMode _stackMode;

    void CalculateAligningGrid( uint32_t bitmapIndex );

public:
    /// creates an instance with the given images
    BaseStacker( const std::vector<Pipeline>& pipelines, StackMode stackMode );

    /// detects stars in the images
    void Registrate();
    /// detects stars and stacks images in one time
    virtual std::shared_ptr<IBitmap>  RegistrateAndStack() = 0;
    /// stacks registered images
    virtual std::shared_ptr<IBitmap> Stack() = 0;

    double GetThreshold() const { return _threshold; }
    void SetThreshold( double threshold ) { _threshold = threshold; };
    double GetMinStarSize() const { return _minStarSize; }
    void SetMinStarSize( uint32_t minStarSize ) { _minStarSize = minStarSize; };
    double GetMaxStarSize() const { return _threshold; }
    void SetMaxStarSize( uint32_t maxStarSize ) { _maxStarSize = maxStarSize; };

    /// needed for compatibility with pipeline API
    IBitmapPtr ProcessBitmap( IBitmapPtr pSrcBitmap = nullptr );
};

ACMB_NAMESPACE_END