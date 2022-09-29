#define _USE_MATH_DEFINES

#include "stacker.h"
#include "FastAligner.h"
#include "registrator.h"

#include "../Codecs/imagedecoder.h"
#include "../Geometry/delaunator.hpp"
#include "../Transforms/deaberratetransform.h"
#include "../Transforms/BitmapSubtractor.h"

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

ACMB_NAMESPACE_BEGIN


template<PixelFormat pixelFormat>
class AddingBitmapHelper
{
    static constexpr uint32_t channelCount = ChannelCount( pixelFormat );

    Stacker& _stacker;
    std::shared_ptr<Bitmap<pixelFormat>> _pBitmap;

    AddingBitmapHelper( Stacker& stacker, std::shared_ptr<Bitmap<pixelFormat>> pBitmap )
        : _stacker( stacker )
        , _pBitmap( pBitmap )
    {

    }

    void Job( uint32_t i )
    {
        using ChannelType = typename PixelFormatTraits<pixelFormat>::ChannelType;

        for ( uint32_t j = 0; j < _stacker._width * channelCount; ++j )
        {
            const auto index = i * _stacker._width * channelCount + j;
            auto& mean = _stacker._means[index];
            auto& dev = _stacker._devs[index];
            auto& n = _stacker._counts[index];
            auto& channel = _pBitmap->GetScanline( 0 )[index];

            const auto sigma = sqrt( dev );
            const auto kappa = 3.0;

            if ( n <= 5 || fabs( mean - channel ) < kappa * sigma )
            {
                dev = n * ( dev + ( channel - mean ) * ( channel - mean ) / ( n + 1 ) ) / ( n + 1 );
                mean = std::clamp( ( n * mean + channel ) / ( n + 1 ), 0.0f, static_cast< float >( std::numeric_limits<ChannelType>::max() ) );
                ++n;
            }
        }
    }

public:

    static void Run( Stacker& stacker, std::shared_ptr<Bitmap<pixelFormat>> pBitmap )
    {
        AddingBitmapHelper helper( stacker, pBitmap );
        oneapi::tbb::parallel_for( oneapi::tbb::blocked_range<int>( 0, pBitmap->GetHeight() ), [&helper] ( const oneapi::tbb::blocked_range<int>& range )
        {
            for ( int i = range.begin(); i < range.end(); ++i )
            {
                helper.Job( i );
            }
        } );
    }
};

template<PixelFormat pixelFormat>
class AddingBitmapWithAlignmentHelper
{
    static constexpr uint32_t channelCount = ChannelCount( pixelFormat );

    Stacker& _stacker;
    std::shared_ptr<Bitmap<pixelFormat>> _pBitmap;

    AddingBitmapWithAlignmentHelper( Stacker& stacker, std::shared_ptr<Bitmap<pixelFormat>> pBitmap )
        : _stacker( stacker )
        , _pBitmap( pBitmap )
    {

    }

    void Job( uint32_t i )
    {
        Stacker::TriangleTransformPair lastPair;

        for ( uint32_t x = 0; x < _stacker._width; ++x )
        {
            PointF p{ static_cast< double >( x ), static_cast< double >( i ) };

            size_t hGridIndex = x / _stacker.gridSize;
            size_t vGridIndex = i / _stacker.gridSize;

            if ( !_stacker._grid.empty() )
            {
                _stacker.ChooseTriangle( p, lastPair, _stacker._grid[vGridIndex * _stacker._gridWidth + hGridIndex] );
                lastPair.second.transform( &p.x, &p.y );
            }


            if ( ( _stacker._grid.empty() || lastPair.second != agg::trans_affine_null() ) && p.x >= 0 && p.x <= _stacker._width - 1 && p.y >= 0 && p.y <= _stacker._height - 1 )
            {
                for ( uint32_t ch = 0; ch < channelCount; ++ch )
                {
                    const auto interpolatedChannel = _pBitmap->GetInterpolatedChannel( static_cast< float >( p.x ), static_cast< float >( p.y ), ch );
                    const size_t index = i * _stacker._width * channelCount + x * channelCount + ch;
                    auto& mean = _stacker._means[index];
                    auto& dev = _stacker._devs[index];
                    auto& n = _stacker._counts[index];

                    auto sigma = sqrt( dev );
                    const auto kappa = 3.0;

                    if ( n <= 5 || fabs( mean - interpolatedChannel ) < kappa * sigma )
                    {
                        dev = n * ( dev + ( interpolatedChannel - mean ) * ( interpolatedChannel - mean ) / ( n + 1 ) ) / ( n + 1 );

                        mean = std::clamp( ( n * mean + interpolatedChannel ) / ( n + 1 ), 0.0f, static_cast< float >( std::numeric_limits<typename PixelFormatTraits<pixelFormat>::ChannelType>::max() ) );
                        ++n;
                    }
                }
            }
        }
    }

public:

    static void Run( Stacker& stacker, std::shared_ptr<Bitmap<pixelFormat>> pBitmap )
    {
        AddingBitmapWithAlignmentHelper helper( stacker, pBitmap );
        oneapi::tbb::parallel_for( oneapi::tbb::blocked_range<int>( 0, pBitmap->GetHeight() ), [&helper] ( const oneapi::tbb::blocked_range<int>& range )
        {
            for ( int i = range.begin(); i < range.end(); ++i )
            {
                helper.Job( i );
            }
        } );
    }
};

class AlignmentHelper
{
    Stacker& _stacker;
    size_t _alignerIndex;
    std::mutex _mutex;

    AlignmentHelper( Stacker& stacker, size_t alignerIndex )
        : _stacker( stacker )
        , _alignerIndex( alignerIndex )
    {
        if ( _stacker._aligners.size() <= _alignerIndex )
            throw std::invalid_argument( "no aligner" );
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
    static void Run( Stacker& stacker, size_t alignerIndex )
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

template<PixelFormat pixelFormat>
class GeneratingResultHelper
{
    Stacker& _stacker;
    std::shared_ptr<Bitmap<pixelFormat>> _pBitmap;

    GeneratingResultHelper( Stacker& stacker )
        : _stacker( stacker )
        , _pBitmap( std::make_shared<Bitmap<pixelFormat>>( stacker._width, stacker._height ) )
    {

    }

    using ChannelType = typename PixelFormatTraits<pixelFormat>::ChannelType;

    void Job( uint32_t i )
    {
        using ChannelType = typename PixelFormatTraits<pixelFormat>::ChannelType;

        ChannelType* pChannel = &_pBitmap->GetScanline( i )[0];
        float* pMean = &_stacker._means[i * _stacker._width * ChannelCount( pixelFormat )];

        for ( uint32_t x = 0; x < _stacker._width; ++x )
            for ( uint32_t ch = 0; ch < ChannelCount( pixelFormat ); ++ch )
            {
                *pChannel = FastRound<ChannelType>( *pMean );
                ++pMean;
                ++pChannel;
            }
    }

public:
    static void Run( Stacker& stacker, std::shared_ptr<Bitmap<pixelFormat>> pBitmap )
    {
        GeneratingResultHelper helper( stacker );
        oneapi::tbb::parallel_for( oneapi::tbb::blocked_range<int>( 0, pBitmap->GetHeight() ), [&helper] ( const oneapi::tbb::blocked_range<int>& range )
        {
            for ( int i = range.begin(); i < range.end(); ++i )
            {
                helper.Job( i );
            }
        } );
        pBitmap->SetData( helper._pBitmap->GetData() );
    }
};

static constexpr bool enableLogging = false;
void Log(const std::string& message)
{
if (enableLogging)
    std::cout << message << std::endl;
}

Stacker::Stacker( const std::vector<Pipeline>& pipelines )
{
    for (const auto& pipeline : pipelines )
    {
        _stackingData.push_back({ pipeline, {}, {} });
    }

    if ( pipelines.empty() )
        throw std::invalid_argument( "no pipelines" );
    
    auto finalParams = _stackingData[0].pipeline.GetFinalParams();
    if ( !finalParams )
        throw std::invalid_argument( "cannot calculate image params" );

    _width = finalParams->GetWidth();
    _height = finalParams->GetHeight();
    _pixelFormat = finalParams->GetPixelFormat();
    _gridWidth = _width / gridSize + ((_width % gridSize) ? 1 : 0);
    _gridHeight = _height / gridSize + ((_height % gridSize) ? 1 : 0);    
}

void Stacker::SetDarkFrame( IBitmapPtr pDarkFrame )
{
    _pDarkFrame = pDarkFrame;
}

IBitmapPtr Stacker::ProcessBitmap( IBitmapPtr pSrcBitmap )
{
    return _doAlignment ? RegistrateAndStack() : Stack();
}

void Stacker::Registrate()
{
    auto pRegistrator = std::make_unique<Registrator>(_threshold, _minStarSize, _maxStarSize);
    for (auto& dsPair : _stackingData)
    {
        auto pBitmap = dsPair.pipeline.RunAndGetBitmap();
        if ( _pDarkFrame )
            BitmapSubtractor::Subtract( pBitmap, _pDarkFrame );       
        
        pRegistrator->Registrate(pBitmap);
        dsPair.stars = pRegistrator->GetStars();

        for (const auto& starVector : dsPair.stars)
        {
            dsPair.totalStarCount += starVector.size();
        }

        Log(dsPair.pipeline.GetFileName() + " is registered");
        Log(std::to_string(dsPair.totalStarCount) + " stars are found");
    }   

    //std::sort(std::begin(_stackingData), std::end(_stackingData), [](const auto& a, const auto& b) { return a.stars.size() > b.stars.size(); });
}

std::shared_ptr<IBitmap> Stacker::Stack()
{
    if (_stackingData.size() == 0)
        return nullptr;
    
    Log(_stackingData[0].pipeline.GetFileName() + " in process");

    auto pRefBitmap = _stackingData[0].pipeline.RunAndGetBitmap();
    if ( _pDarkFrame )
        BitmapSubtractor::Subtract( pRefBitmap, _pDarkFrame );

    Log( _stackingData[0].pipeline.GetFileName() + " is read" );   

    if (_stackingData.size() == 1)
        return pRefBitmap;

    const auto& refStars = _stackingData[0].stars;

    if (_doAlignment)
    {
        _aligners.clear();

        for (const auto& refStarVector : refStars)
            _aligners.push_back(std::make_shared<FastAligner>(refStarVector));
    }
    
    _means.resize(_width  * _height * ChannelCount(pRefBitmap->GetPixelFormat()));
    _devs.resize(_width * _height * ChannelCount(pRefBitmap->GetPixelFormat()));
    _counts.resize(_width * _height * ChannelCount(pRefBitmap->GetPixelFormat()));

    CALL_HELPER(AddingBitmapHelper, pRefBitmap);

    Log( _stackingData[0].pipeline.GetFileName() + " is stacked" );

    for (uint32_t i = 1; i < _stackingData.size(); ++i)
    {
        Log( _stackingData[i].pipeline.GetFileName() + " in process" );
        auto pTargetBitmap = _stackingData[i].pipeline.RunAndGetBitmap();
        Log( _stackingData[i].pipeline.GetFileName() + " is read" );
        if ( _pDarkFrame )
            BitmapSubtractor::Subtract( pTargetBitmap, _pDarkFrame );

        if (pRefBitmap->GetPixelFormat() != pTargetBitmap->GetPixelFormat())
            throw std::runtime_error("bitmaps in stack should have the same pixel format");      

        if (!_doAlignment)
        {
            CALL_HELPER(AddingBitmapHelper, pTargetBitmap);
            continue;
        }

        
        StackWithAlignment(pTargetBitmap, i);
    }
    
    auto pRes = IBitmap::Create(_width, _height, pRefBitmap->GetPixelFormat());

    CALL_HELPER(GeneratingResultHelper, pRes);

    return pRes;
}

void Stacker::StackWithAlignment(IBitmapPtr pTargetBitmap, uint32_t i)
{
    _matches.clear();
    AlignmentHelper::Run(*this, i);
    Log(std::to_string(_matches.size()) + " matching stars");

    std::vector<double> coords;
    for (auto& match : _matches)
    {
        coords.push_back(match.first.x);
        coords.push_back(match.first.y);
    }

    delaunator::Delaunator d(coords);

    Grid grid;
    _grid.clear();
    _grid.resize(_gridWidth * _gridHeight);

    for (std::size_t i = 0; i < d.triangles.size(); i += 3)
    {
        Triangle targetTriangle{ PointF {d.coords[2 * d.triangles[i]], d.coords[2 * d.triangles[i] + 1]}, PointF {d.coords[2 * d.triangles[i + 1]], d.coords[2 * d.triangles[i + 1] + 1]}, PointF {d.coords[2 * d.triangles[i + 2]], d.coords[2 * d.triangles[i + 2] + 1]} };
        Triangle refTriangle{ _matches[targetTriangle.vertices[0]], _matches[targetTriangle.vertices[1]], _matches[targetTriangle.vertices[2]] };

        TriangleTransformPair pair = { refTriangle, agg::trans_affine(reinterpret_cast<double*>(refTriangle.vertices.data()), reinterpret_cast<double*>(targetTriangle.vertices.data())) };

        for (size_t j = 0; j < _gridWidth * _gridHeight; ++j)
        {
            RectF cell =
            {
                static_cast<double>((j % _gridWidth) * gridSize),
                static_cast<double>((j / _gridWidth) * gridSize),
                gridSize,
                gridSize
            };

            if (refTriangle.GetBoundingBox().Overlaps(cell))
            {
                _grid[j].push_back(pair);
            }
        }
    }

    Log( _stackingData[i].pipeline.GetFileName() + " grid is calculated" );

    CALL_HELPER(AddingBitmapWithAlignmentHelper, pTargetBitmap);

    Log( _stackingData[i].pipeline.GetFileName() + " is stacked" );
}

std::shared_ptr<IBitmap>  Stacker::RegistrateAndStack()
{
    if (_stackingData.size() == 0)
        return nullptr;   

    auto pRefBitmap = _stackingData[0].pipeline.RunAndGetBitmap();
    if ( _pDarkFrame )
        BitmapSubtractor::Subtract( pRefBitmap, _pDarkFrame );    

    if (_stackingData.size() == 1)
        return pRefBitmap;

    auto pRegistrator = std::make_unique<Registrator>(_threshold, _minStarSize, _maxStarSize);

    pRegistrator->Registrate(pRefBitmap);
    _stackingData[0].stars = pRegistrator->GetStars();

    const auto& refStars = _stackingData[0].stars;

     _aligners.clear();
     for (const auto& refStarVector : refStars)
         _aligners.push_back(std::make_shared<FastAligner>(refStarVector));    

     _means.resize(_width * _height * ChannelCount(pRefBitmap->GetPixelFormat()));
     _devs.resize(_width * _height * ChannelCount(pRefBitmap->GetPixelFormat()));
     _counts.resize(_width * _height * ChannelCount(pRefBitmap->GetPixelFormat()));

    CALL_HELPER(AddingBitmapHelper, pRefBitmap);

    Log( _stackingData[0].pipeline.GetFileName() + " is stacked" );

    for (uint32_t i = 1; i < _stackingData.size(); ++i)
    {
        Log( _stackingData[i].pipeline.GetFileName() + " in process" );
        auto pTargetBitmap = _stackingData[i].pipeline.RunAndGetBitmap();
        Log( _stackingData[i].pipeline.GetFileName() + " bitmap is read" );
        if ( _pDarkFrame )
            BitmapSubtractor::Subtract( pTargetBitmap, _pDarkFrame );

        if (pRefBitmap->GetPixelFormat() != pTargetBitmap->GetPixelFormat())
            throw std::runtime_error("bitmaps in stack should have the same pixel format");       

        pRegistrator->Registrate(pTargetBitmap);
        _stackingData[i].stars = pRegistrator->GetStars();

        StackWithAlignment(pTargetBitmap, i);
    }

    auto pRes = IBitmap::Create(_width, _height, pRefBitmap->GetPixelFormat());
    CALL_HELPER(GeneratingResultHelper, pRes);    

    return pRes;    
}

void Stacker::ChooseTriangle(PointF p, std::pair<Triangle, agg::trans_affine>& lastPair, const Stacker::GridCell& trianglePairs)
{
    if (lastPair.first.IsPointInside(p))
        return;

    double minDist = std::numeric_limits<double>::max();
    TriangleTransformPair nearest;

    for (const auto& pair : trianglePairs)
    {
        if (pair.first.IsPointInside(p))
        {
            lastPair = pair;
            return;
        }

        double dist = p.Distance(pair.first.GetCenter());
        if (dist < minDist)
        {
            nearest = pair;
            minDist = dist;
        }
    }

    lastPair = nearest;
}

ACMB_NAMESPACE_END
