#pragma once
#include "FastAligner.h"
#include "./../AGG/agg_trans_affine.h"
#include "../Core/pipeline.h"
#include "../Geometry/triangle.h"

ACMB_NAMESPACE_BEGIN

class Registrator;

struct StackingDatum
{
    Pipeline pipeline;
    std::vector<std::vector<Star>> stars;
    uint64_t totalStarCount;
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

    uint32_t _gridWidth = 0;
    uint32_t _gridHeight = 0;

    std::vector<std::shared_ptr<FastAligner>> _aligners;
    MatchMap _matches;

    double _threshold = 25.0;
    uint32_t _minStarSize = 5;
    uint32_t _maxStarSize = 25;

    StackMode _stackMode;

    void CalculateAligningGrid( const std::vector<std::vector<Star>>& stars  );

public:
    /// creates an instance with the given images
    BaseStacker( const std::vector<Pipeline>& pipelines, StackMode stackMode );

    /// creates an instance with one image
    BaseStacker( const ImageParams& imageParams, StackMode stackMode );

    /// detects stars in the images
    void Registrate();
    /// detects stars and stacks images in one time
    std::shared_ptr<IBitmap>  RegistrateAndStack();
    /// stacks registered images
    std::shared_ptr<IBitmap> Stack();

    void AddBitmap( Pipeline& pipeline );
    void AddBitmap( std::shared_ptr<IBitmap> pBitmap );
    std::shared_ptr<IBitmap> GetResult();

    virtual void CallAddBitmapHelper( std::shared_ptr<IBitmap> pBitmap ) = 0;
    virtual void CallAddBitmapWithAlignmentHelper( std::shared_ptr<IBitmap> pBitmap ) = 0;
    virtual std::shared_ptr<IBitmap> CallGeneratingResultHelper() = 0;

    double GetThreshold() const { return _threshold; }
    void SetThreshold( double threshold ) { _threshold = threshold; };
    double GetMinStarSize() const { return _minStarSize; }
    void SetMinStarSize( uint32_t minStarSize ) { _minStarSize = minStarSize; };
    double GetMaxStarSize() const { return _threshold; }
    void SetMaxStarSize( uint32_t maxStarSize ) { _maxStarSize = maxStarSize; };

    /// needed for compatibility with pipeline API
    std::shared_ptr<IBitmap> ProcessBitmap( std::shared_ptr<IBitmap> pSrcBitmap = nullptr );
};

ACMB_NAMESPACE_END
