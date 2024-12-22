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

class IStacker : public IPipelineFirstElement
{
protected:
    std::vector<StackingDatum> _stackingData;
    double _threshold = 25.0;
    uint32_t _minStarSize = 5;
    uint32_t _maxStarSize = 25;

public:
    /// creates an instance without images (only with image parameters)
    IStacker( const ImageParams& imageParams );

    virtual ~IStacker() = default;    
    
    virtual void AddBitmap( Pipeline pipeline );
    virtual std::shared_ptr<IBitmap> GetResult() = 0;

    void ValidateFrameParams( const ImageParams& imageParams );

    /// detects stars and stacks images in one time
    //std::shared_ptr<IBitmap>  RegistrateAndStack();
    /// stacks registered images
    void AddBitmap( std::shared_ptr<IBitmap> pBitmap );

    void AddBitmaps( const std::vector<std::shared_ptr<IBitmap>>& bitmaps );
    void AddBitmaps( const std::vector<Pipeline>& pipelines );    

    double GetThreshold() const { return _threshold; }
    void SetThreshold( double threshold ) { _threshold = threshold; };
    double GetMinStarSize() const { return _minStarSize; }
    void SetMinStarSize( uint32_t minStarSize ) { _minStarSize = minStarSize; };
    double GetMaxStarSize() const { return _threshold; }
    void SetMaxStarSize( uint32_t maxStarSize ) { _maxStarSize = maxStarSize; };


    /// needed for compatibility with pipeline API
    virtual std::shared_ptr<IBitmap> ProcessBitmap( std::shared_ptr<IBitmap> pSrcBitmap = nullptr );
};

class SimpleStacker : public IStacker
{
    std::vector<float> _means;
    std::vector<float> _devs;
    std::vector<uint16_t> _counts;

    std::unique_ptr<FastAligner> aligner_;

public:
    //SimpleStacker( const std::vector<Pipeline>& pipelines );
    SimpleStacker( const ImageParams& imageParams );

    using IStacker::AddBitmap;
    void AddBitmap( Pipeline pipeline ) override;
    std::shared_ptr<IBitmap> GetResult() override;
};

class BaseStacker : public IStacker
{
    friend class AlignmentHelper;

public:
    using TriangleTransformPair = std::pair<Triangle, agg::trans_affine>;
    using GridCell = std::vector<TriangleTransformPair>;
    using Grid = std::vector<GridCell>;

protected:
    Grid _grid;

    uint32_t _gridWidth = 0;
    uint32_t _gridHeight = 0;

    std::vector<std::shared_ptr<FastAligner>> _aligners;
    MatchMap _matches;

    StackMode _stackMode;

    void CalculateAligningGrid( const std::vector<std::vector<Star>>& stars  );

public:
    /// creates an instance with the given images
    //BaseStacker( const std::vector<Pipeline>& pipelines, StackMode stackMode );

    /// creates an instance with one image
    BaseStacker( const ImageParams& imageParams, StackMode stackMode );

    /// stacks registered images
    ///virtual std::shared_ptr<IBitmap> Stack() override;

    using IStacker::AddBitmap;
    void AddBitmap( Pipeline pipeline );
    virtual std::shared_ptr<IBitmap> GetResult() override;

    virtual void CallAddBitmapHelper( std::shared_ptr<IBitmap> pBitmap ) = 0;
    virtual void CallAddBitmapWithAlignmentHelper( std::shared_ptr<IBitmap> pBitmap ) = 0;
    virtual std::shared_ptr<IBitmap> CallGeneratingResultHelper() = 0;

    std::shared_ptr<IBitmap> ProcessBitmap( std::shared_ptr<IBitmap> pSrcBitmap = nullptr ) override;

};

ACMB_NAMESPACE_END
