#include "BaseStacker.h"
#include "registrator.h"
#include "StackEngineConstants.h"
#include "../Core/log.h"
#include "../Geometry/delaunator.hpp"
#include "../Transforms/DebayerTransform.h"
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

ACMB_NAMESPACE_BEGIN

class AlignmentHelper
{
    std::vector<std::shared_ptr<FastAligner>>& aligners_;
    const std::vector<std::vector<Star>>& stars_;
    MatchMap& matches_;
    std::mutex _mutex;

    AlignmentHelper( std::vector<std::shared_ptr<FastAligner>>& aligners, const std::vector<std::vector<Star>>& stars, MatchMap& matches )
        : aligners_( aligners )
        , stars_( stars )
        , matches_( matches )
    {        
    }

    void Job( uint32_t i )
    {
        aligners_[i]->Align( stars_[i] );
        auto tileMatches = aligners_[i]->GetMatches();

        std::lock_guard<std::mutex> lock( _mutex );
        matches_.insert( tileMatches.begin(), tileMatches.end() );
    }

public:
    static void Run( std::vector<std::shared_ptr<FastAligner>>& aligners, const std::vector<std::vector<Star>>& stars, MatchMap& matches, uint32_t width, uint32_t height )
    {
        AlignmentHelper helper( aligners, stars, matches );
        auto [hTileCount, vTileCount] = GetTileCounts( width, height );
        oneapi::tbb::parallel_for( oneapi::tbb::blocked_range<int>( 0, hTileCount * vTileCount ), [&helper] ( const oneapi::tbb::blocked_range<int>& range )
        {
            for ( int i = range.begin(); i < range.end(); ++i )
            {
                helper.Job( i );
            }
        } );
    }
};

BaseStacker::BaseStacker( const std::vector<Pipeline>& pipelines, StackMode stackMode )
    : _stackMode( stackMode )
{
    for ( const auto& pipeline : pipelines )
    {
        _stackingData.push_back( { pipeline, {}, {} } );
        if ( ( stackMode == StackMode::Light || stackMode == StackMode::LightNoAlign ) && pipeline.GetFinalParams()->GetPixelFormat() == PixelFormat::Bayer16 )
            _stackingData.back().pipeline.AddTransform<DebayerTransform>( pipeline.GetCameraSettings() );
    }

    if ( pipelines.empty() )
        throw std::invalid_argument( "no pipelines" );

    auto finalParams = _stackingData[0].pipeline.GetFinalParams();
    if ( !finalParams )
        throw std::invalid_argument( "cannot calculate image params" );

    _width = finalParams->GetWidth();
    _height = finalParams->GetHeight();
    _pixelFormat = finalParams->GetPixelFormat();
    if ( _stackMode == StackMode::Light && _pixelFormat == PixelFormat::Bayer16 )
        _pixelFormat = PixelFormat::RGB48;

    _gridWidth = _width / cGridPixelSize + ( ( _width % cGridPixelSize ) ? 1 : 0 );
    _gridHeight = _height / cGridPixelSize + ( ( _height % cGridPixelSize ) ? 1 : 0 );
}

BaseStacker::BaseStacker( const ImageParams& imageParams, StackMode stackMode )
: _stackMode(stackMode)
{
    _width = imageParams.GetWidth();
    _height = imageParams.GetHeight();

    _pixelFormat = imageParams.GetPixelFormat();
    if ( _stackMode == StackMode::Light && _pixelFormat == PixelFormat::Bayer16 )
        _pixelFormat = PixelFormat::RGB48;

    _gridWidth = _width / cGridPixelSize + ( ( _width % cGridPixelSize ) ? 1 : 0 );
    _gridHeight = _height / cGridPixelSize + ( ( _height % cGridPixelSize ) ? 1 : 0 );
}

void BaseStacker::Registrate()
{
    auto pRegistrator = std::make_unique<Registrator>( _threshold, _minStarSize, _maxStarSize );
    for ( auto& dsPair : _stackingData )
    {
        auto pBitmap = dsPair.pipeline.RunAndGetBitmap();

        pRegistrator->Registrate( pBitmap );
        dsPair.stars = pRegistrator->GetStars();

        for ( const auto& starVector : dsPair.stars )
        {
            dsPair.totalStarCount += starVector.size();
        }

        Log( dsPair.pipeline.GetFileName() + " is registered" );
        Log( std::to_string( dsPair.totalStarCount ) + " stars are found" );
    }

    //std::sort(std::begin(_stackingData), std::end(_stackingData), [](const auto& a, const auto& b) { return a.stars.size() > b.stars.size(); });
}

void BaseStacker::CalculateAligningGrid( const std::vector<std::vector<Star>>& stars )
{
    _matches.clear();
    AlignmentHelper::Run( _aligners, stars, _matches, _width, _height );

    Log( std::to_string( _matches.size() ) + " matching stars" );

    std::vector<double> coords;
    for ( auto& match : _matches )
    {
        coords.push_back( match.first.x );
        coords.push_back( match.first.y );
    }

    delaunator::Delaunator d( coords );

    Grid grid;
    _grid.clear();
    _grid.resize( _gridWidth * _gridHeight );

    for ( std::size_t i = 0; i < d.triangles.size(); i += 3 )
    {
        Triangle targetTriangle{ PointF {d.coords[2 * d.triangles[i]], d.coords[2 * d.triangles[i] + 1]}, PointF {d.coords[2 * d.triangles[i + 1]], d.coords[2 * d.triangles[i + 1] + 1]}, PointF {d.coords[2 * d.triangles[i + 2]], d.coords[2 * d.triangles[i + 2] + 1]} };
        Triangle refTriangle{ _matches[targetTriangle.vertices[0]], _matches[targetTriangle.vertices[1]], _matches[targetTriangle.vertices[2]] };

        TriangleTransformPair pair = { refTriangle, agg::trans_affine( reinterpret_cast< double* >( refTriangle.vertices.data() ), reinterpret_cast< double* >( targetTriangle.vertices.data() ) ) };

        for ( size_t j = 0; j < _gridWidth * _gridHeight; ++j )
        {
            RectF cell =
            {
                static_cast< double >( ( j % _gridWidth ) * cGridPixelSize ),
                static_cast< double >( ( j / _gridWidth ) * cGridPixelSize ),
                cGridPixelSize,
                cGridPixelSize
            };

            if ( refTriangle.GetBoundingBox().Overlaps( cell ) )
            {
                _grid[j].push_back( pair );
            }
        }
    }
}

std::shared_ptr<IBitmap> BaseStacker::Stack()
{
    if (_stackingData.size() == 0)
        return nullptr;

    Log(_stackingData[0].pipeline.GetFileName() + " in process");

    auto pRefBitmap = _stackingData[0].pipeline.RunAndGetBitmap();

    Log( _stackingData[0].pipeline.GetFileName() + " is read" );

    if (_stackingData.size() == 1)
        return pRefBitmap;

    const auto& refStars = _stackingData[0].stars;

    if ( _stackMode == StackMode::Light )
    {
        _aligners.clear();

        for (const auto& refStarVector : refStars)
            _aligners.push_back(std::make_shared<FastAligner>(refStarVector));
    }

    CallAddBitmapHelper( pRefBitmap );

    Log( _stackingData[0].pipeline.GetFileName() + " is stacked" );

    for (uint32_t i = 1; i < _stackingData.size(); ++i)
    {
        AddBitmap( _stackingData[i].pipeline );
    }

    return CallGeneratingResultHelper();
}

void BaseStacker::AddBitmap(Pipeline& pipeline)
{
    Log( pipeline.GetFileName() + " in process" );
    if ( ( _stackMode == StackMode::Light || _stackMode == StackMode::LightNoAlign ) && pipeline.GetFinalParams()->GetPixelFormat() == PixelFormat::Bayer16 )
        pipeline.AddTransform<DebayerTransform>( pipeline.GetCameraSettings() );

    auto pBitmap = pipeline.RunAndGetBitmap();
    Log( pipeline.GetFileName() + " is read" );
    if ( _stackMode != StackMode::Light )
    {
        CallAddBitmapHelper( pBitmap );
        return;
    }

    auto pRegistrator = std::make_shared<Registrator>(_threshold, _minStarSize, _maxStarSize);
    pRegistrator->Registrate( pBitmap );
    const auto& stars = pRegistrator->GetStars();

    if ( _aligners.empty() )
    {
        for (const auto& starVector : stars)
                 _aligners.push_back(std::make_shared<FastAligner>(starVector));

        CallAddBitmapHelper( pBitmap );
        Log( pipeline.GetFileName() + " is stacked" );
        return;
    }

    CalculateAligningGrid( stars );
    Log( pipeline.GetFileName() + " grid is calculated" );
    CallAddBitmapWithAlignmentHelper( pBitmap );
    Log( pipeline.GetFileName() + " is stacked" );
}

IBitmapPtr BaseStacker::GetResult()
{
    return CallGeneratingResultHelper();
}

std::shared_ptr<IBitmap>  BaseStacker::RegistrateAndStack()
{
    if (_stackingData.size() == 0)
        return nullptr;

    for ( uint32_t i = 0; i < _stackingData.size(); ++i )
    {
        AddBitmap( _stackingData[i].pipeline );
    }

    return CallGeneratingResultHelper();
}

IBitmapPtr BaseStacker::ProcessBitmap( IBitmapPtr )
{
    return ( _stackMode == StackMode::Light ) ? RegistrateAndStack() : Stack();
}

ACMB_NAMESPACE_END
