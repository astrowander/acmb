#define _USE_MATH_DEFINES

#include "stacker.h"
#include "registrator.h"
#include "../Core/log.h"
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#define CALL_HELPER(Helper, pBitmap) \
    switch (pBitmap->GetPixelFormat()) \
    { \
        case PixelFormat::Gray8: \
            Helper<PixelFormat::Gray8>::Run(*this, std::static_pointer_cast<Bitmap<PixelFormat::Gray8>>(pBitmap)); \
            break; \
        case PixelFormat::Gray16:\
             Helper<PixelFormat::Gray16>::Run(*this, std::static_pointer_cast<Bitmap<PixelFormat::Gray16>>(pBitmap)); \
            break; \
        case PixelFormat::RGB24:\
             Helper<PixelFormat::RGB24>::Run(*this, std::static_pointer_cast<Bitmap<PixelFormat::RGB24>>(pBitmap)); \
            break; \
        case PixelFormat::RGB48:\
              Helper<PixelFormat::RGB48>::Run(*this, std::static_pointer_cast<Bitmap<PixelFormat::RGB48>>(pBitmap)); \
            break; \
        case PixelFormat::Bayer16:\
              Helper<PixelFormat::Bayer16>::Run(*this, std::static_pointer_cast<Bitmap<PixelFormat::Bayer16>>(pBitmap)); \
            break; \
        default:\
            throw std::runtime_error("pixel format should be known");}

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

Stacker::Stacker( const std::vector<Pipeline>& pipelines, StackMode stackMode )
: BaseStacker( pipelines, stackMode )
{}

std::shared_ptr<IBitmap> Stacker::Stack()
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
       
        if (pRefBitmap->GetPixelFormat() != pTargetBitmap->GetPixelFormat())
            throw std::runtime_error("bitmaps in stack should have the same pixel format");      

        if ( _stackMode != StackMode::Light )
        {
            CALL_HELPER(AddingBitmapHelper, pTargetBitmap);
            Log( _stackingData[i].pipeline.GetFileName() + " is stacked" );
            continue;
        }

        
        StackWithAlignment(pTargetBitmap, i);
        Log( _stackingData[i].pipeline.GetFileName() + " is stacked" );
    }
    
    auto pRes = IBitmap::Create(_width, _height, pRefBitmap->GetPixelFormat());

    CALL_HELPER(GeneratingResultHelper, pRes);

    return pRes;
}

void Stacker::StackWithAlignment( IBitmapPtr pTargetBitmap, uint32_t bitmapIndex )
{
    CalculateAligningGrid( bitmapIndex );    

    Log( _stackingData[bitmapIndex].pipeline.GetFileName() + " grid is calculated" );

    CALL_HELPER(AddingBitmapWithAlignmentHelper, pTargetBitmap);

    Log( _stackingData[bitmapIndex].pipeline.GetFileName() + " is stacked" );
}

std::shared_ptr<IBitmap>  Stacker::RegistrateAndStack()
{
    if (_stackingData.size() == 0)
        return nullptr;   

    auto pRefBitmap = _stackingData[0].pipeline.RunAndGetBitmap();
   
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
