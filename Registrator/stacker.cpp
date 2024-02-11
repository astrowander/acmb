#define _USE_MATH_DEFINES

#include "stacker.h"
#include "StackEngineConstants.h"
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#define CALL_HELPER(Helper, pBitmap) \
    switch (pBitmap->GetPixelFormat()) \
    { \
        case PixelFormat::Gray8: {\
            auto pGray8Bitmap = std::static_pointer_cast<Bitmap<PixelFormat::Gray8>>(pBitmap);\
            Helper<PixelFormat::Gray8>::Run(*this, pGray8Bitmap); \
            break; }\
        case PixelFormat::Gray16:{\
            auto pGray16Bitmap = std::static_pointer_cast<Bitmap<PixelFormat::Gray16>>(pBitmap);\
            Helper<PixelFormat::Gray16>::Run(*this, pGray16Bitmap); \
            break; }\
        case PixelFormat::RGB24:{\
            auto pRGB24Bitmap = std::static_pointer_cast<Bitmap<PixelFormat::RGB24>>(pBitmap);\
            Helper<PixelFormat::RGB24>::Run(*this, pRGB24Bitmap); \
            break; }\
        case PixelFormat::RGB48:{\
            auto pRGB48Bitmap = std::static_pointer_cast<Bitmap<PixelFormat::RGB48>>(pBitmap);\
            Helper<PixelFormat::RGB48>::Run(*this, pRGB48Bitmap); \
            break; }\
        case PixelFormat::Bayer16:{\
            auto pBayer16Bitmap = std::static_pointer_cast<Bitmap<PixelFormat::Bayer16>>(pBitmap);\
            Helper<PixelFormat::Bayer16>::Run(*this, pBayer16Bitmap); \
            break; }\
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
            if ( _stacker._stackMode == StackMode::StarTrails )
            {
                mean = std::max<float>( mean, _pBitmap->GetScanline( 0 )[index] );
                continue;
            }
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

            size_t hGridIndex = x / cGridPixelSize;
            size_t vGridIndex = i / cGridPixelSize;

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
    static void Run( Stacker& stacker, std::shared_ptr<Bitmap<pixelFormat>>& pBitmap )
    {
        GeneratingResultHelper helper( stacker );
        oneapi::tbb::parallel_for( oneapi::tbb::blocked_range<int>( 0, helper._pBitmap->GetHeight() ), [&helper] ( const oneapi::tbb::blocked_range<int>& range )
        {
            for ( int i = range.begin(); i < range.end(); ++i )
            {
                helper.Job( i );
            }
        } );
        pBitmap = helper._pBitmap;
    }
};

Stacker::Stacker( const std::vector<Pipeline>& pipelines, StackMode stackMode )
: BaseStacker( pipelines, stackMode )
{
    const size_t size = _width * _height * ChannelCount( _pixelFormat );
    _means.resize( size );
    _devs.resize( size );
    _counts.resize( size );
}

Stacker::Stacker( const ImageParams& imageParams, StackMode stackMode )
:BaseStacker(imageParams, stackMode)
{
    const size_t size = _width * _height * ChannelCount( _pixelFormat );
    _means.resize( size );
    _devs.resize( size );
    _counts.resize( size );
}

void Stacker::CallAddBitmapHelper( IBitmapPtr pBitmap )
{
    CALL_HELPER( AddingBitmapHelper, pBitmap);
}

void Stacker::CallAddBitmapWithAlignmentHelper( IBitmapPtr pBitmap )
{
    CALL_HELPER( AddingBitmapWithAlignmentHelper, pBitmap );
}

IBitmapPtr Stacker::CallGeneratingResultHelper()
{
    switch ( _pixelFormat )
    {
        case PixelFormat::Gray8:
        {
            std::shared_ptr<Bitmap<PixelFormat::Gray8>> pBitmap;
            GeneratingResultHelper<PixelFormat::Gray8>::Run( *this, pBitmap );
            return pBitmap;
        }
        case PixelFormat::Gray16:
        {
            std::shared_ptr<Bitmap<PixelFormat::Gray16>> pBitmap;
            GeneratingResultHelper<PixelFormat::Gray16>::Run( *this, pBitmap );
            return pBitmap;
        }
        case PixelFormat::RGB24:
        {
            std::shared_ptr<Bitmap<PixelFormat::RGB24>> pBitmap;
            GeneratingResultHelper<PixelFormat::RGB24>::Run( *this, pBitmap );
            return pBitmap;
        }
        case PixelFormat::RGB48:
        {
            std::shared_ptr<Bitmap<PixelFormat::RGB48>> pBitmap;
            GeneratingResultHelper<PixelFormat::RGB48>::Run( *this, pBitmap );
            return pBitmap;
        }
        case PixelFormat::Bayer16:
        {
            std::shared_ptr<Bitmap<PixelFormat::Bayer16>> pBitmap;
            GeneratingResultHelper<PixelFormat::Bayer16>::Run( *this, pBitmap );
            return pBitmap;
        }
        default:
            throw std::runtime_error( "pixel format should be specified" );
    }
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
