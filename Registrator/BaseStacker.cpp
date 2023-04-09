#include "BaseStacker.h"
#include "registrator.h"
#include "../Core/log.h"
#include "../Geometry/delaunator.hpp"
#include "../Transforms/DebayerTransform.h"
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

ACMB_NAMESPACE_BEGIN

class AlignmentHelper
{
    BaseStacker& _stacker;
    size_t _alignerIndex;
    std::mutex _mutex;

    AlignmentHelper( BaseStacker& stacker, size_t alignerIndex )
        : _stacker( stacker )
        , _alignerIndex( alignerIndex )
    {
        if ( _stacker._stackingData.size() <= _alignerIndex )
            throw std::invalid_argument( "aligner index exceeds tile count" );
    }

    void Job( uint32_t i )
    {
        _stacker._aligners[i]->Align( _stacker._stackingData[_alignerIndex].stars[i] );
        auto tileMatches = _stacker._aligners[i]->GetMatches();

        _mutex.lock();
        _stacker._matches.insert( tileMatches.begin(), tileMatches.end() );
        _mutex.unlock();
    }

public:
    static void Run( BaseStacker& stacker, size_t alignerIndex )
    {
        AlignmentHelper helper( stacker, alignerIndex );
        auto [hTileCount, vTileCount] = GetTileCounts( stacker._width, stacker._height );
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
    _gridWidth = _width / gridSize + ( ( _width % gridSize ) ? 1 : 0 );
    _gridHeight = _height / gridSize + ( ( _height % gridSize ) ? 1 : 0 );
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

void BaseStacker::CalculateAligningGrid( uint32_t bitmapIndex )
{
    _matches.clear();
    AlignmentHelper::Run( *this, bitmapIndex );

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
                static_cast< double >( ( j % _gridWidth ) * gridSize ),
                static_cast< double >( ( j / _gridWidth ) * gridSize ),
                gridSize,
                gridSize
            };

            if ( refTriangle.GetBoundingBox().Overlaps( cell ) )
            {
                _grid[j].push_back( pair );
            }
        }
    }
}

IBitmapPtr BaseStacker::ProcessBitmap( IBitmapPtr )
{
    return ( _stackMode == StackMode::Light ) ? RegistrateAndStack() : Stack();
}

ACMB_NAMESPACE_END