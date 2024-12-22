#include "BaseStacker.h"
#include "registrator.h"
#include "StackEngineConstants.h"
#include "../Core/log.h"
#include "../Geometry/delaunator.hpp"
#include "../Transforms/DebayerTransform.h"
#include "../Transforms/AffineTransform.h"
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

IStacker::IStacker( const ImageParams& imageParams )
{
    _width = imageParams.GetWidth();
    _height = imageParams.GetHeight();
    _pixelFormat = imageParams.GetPixelFormat();
}

void IStacker::ValidateFrameParams( const ImageParams& imageParams )
{
    if ( imageParams.GetHeight() != _height ||
         imageParams.GetWidth() != _width ||
         imageParams.GetPixelFormat() != _pixelFormat )
    {
        throw std::invalid_argument( "incompatible image params" );
    }
}

void IStacker::AddBitmap( Pipeline pipeline )
{
    auto finalParams = pipeline.GetFinalParams();
    if ( !finalParams )
        throw std::invalid_argument( "cannot calculate image params" );

    ValidateFrameParams( *finalParams );
}

void IStacker::AddBitmap( std::shared_ptr<IBitmap> pBitmap )
{
    if ( !pBitmap )
        throw std::invalid_argument( "no bitmap" );

    if ( !_pCameraSettings )
        _pCameraSettings = pBitmap->GetCameraSettings();

    Pipeline pipeline{ pBitmap };
    return AddBitmap( pipeline );
}

void IStacker::AddBitmaps( const std::vector<std::shared_ptr<IBitmap>>& pBitmaps )
{
    for ( auto pBitmap : pBitmaps )
    {
        AddBitmap( pBitmap );
    }
}

void IStacker::AddBitmaps( const std::vector<Pipeline>& pipelines )
{
    for ( auto& pipeline : pipelines )
    {
        AddBitmap( pipeline );
    }
}

/*IBitmapPtr IStacker::Stack()
{
    if ( _stackingData.size() == 0 )
        return nullptr;

    for ( uint32_t i = 0; i < _stackingData.size(); ++i )
    {
        AddBitmap( _stackingData[i].pipeline );
    }

    return GetResult();
}*/

template<PixelFormat pixelFormat>
void AddBitmapInternal( std::shared_ptr <Bitmap <pixelFormat>> pBitmap, std::vector<float>& means, std::vector<float>& devs, std::vector<uint16_t>& counts )
{
if ( !pBitmap )
        throw std::invalid_argument( "no bitmap" );

using ChannelType = typename PixelFormatTraits<pixelFormat>::ChannelType;
static const uint32_t channelCount = PixelFormatTraits<pixelFormat>::channelCount;

oneapi::tbb::parallel_for( oneapi::tbb::blocked_range<uint32_t>( 0u, pBitmap->GetHeight() ), [&]( const oneapi::tbb::blocked_range<uint32_t>& range )
{

    for ( uint32_t i = range.begin(); i < range.end(); ++i )
    {        
        for ( uint32_t j = 0; j < pBitmap->GetWidth() * channelCount; ++j )
        {
            const auto index = i * pBitmap->GetWidth() * channelCount + j;
            auto& mean = means[index];

            auto& dev = devs[index];
            auto& n = counts[index];
            auto& channel = pBitmap->GetScanline( 0 )[index];

            const auto sigma = sqrt( dev );
            const auto kappa = 3.0;

            if ( n <= 5 || fabs( mean - channel ) < kappa * sigma )
            {
                dev = n * (dev + (channel - mean) * (channel - mean) / (n + 1)) / (n + 1);
                mean = std::clamp( (n * mean + channel) / (n + 1), 0.0f, static_cast< float >(std::numeric_limits<ChannelType>::max()) );
                ++n;
            }
        }
    }
} );
}

template<PixelFormat pixelFormat>
std::shared_ptr<Bitmap<pixelFormat>> GetResultInternal(  uint32_t width, uint32_t height, std::vector<float>& means )
{
    static const uint32_t channelCount = PixelFormatTraits<pixelFormat>::channelCount;
    using ChannelType = typename PixelFormatTraits<pixelFormat>::ChannelType;

    auto pBitmap = std::make_shared<Bitmap<pixelFormat>>( width, height );
    oneapi::tbb::parallel_for( oneapi::tbb::blocked_range<uint32_t>( 0u, pBitmap->GetHeight() ), [&] ( const oneapi::tbb::blocked_range<uint32_t>& range )
    {
        for ( uint32_t i = range.begin(); i < range.end(); ++i )
        {
            auto* pScanline = pBitmap->GetScanline( i );
            for ( uint32_t j = 0; j < pBitmap->GetWidth() * channelCount; ++j )
            {
                const auto index = i * pBitmap->GetWidth() * channelCount + j;
                pScanline[j] = FastRound<ChannelType>( means[index] );
            }
        }
    } );

    return pBitmap;
}

SimpleStacker::SimpleStacker( const ImageParams& imageParams )
: IStacker( imageParams )
{
    if ( _pixelFormat == PixelFormat::Bayer16 )
        _pixelFormat = PixelFormat::RGB48;

    const size_t size = _width * _height * ChannelCount( _pixelFormat );
    _means.resize( size );
    _devs.resize( size );
    _counts.resize( size );    
}

void SimpleStacker::AddBitmap( Pipeline pipeline )
{
    IStacker::AddBitmap( pipeline );

    Log( pipeline.GetFileName() + " in process" );
    if ( pipeline.GetFinalParams()->GetPixelFormat() == PixelFormat::Bayer16 )
        pipeline.AddTransform<DebayerTransform>( pipeline.GetCameraSettings() );

    auto pBitmap = pipeline.RunAndGetBitmap();
    Log( pipeline.GetFileName() + " is read" );

    auto pRegistrator = std::make_shared<Registrator>( _threshold, _minStarSize, _maxStarSize );
    pRegistrator->Registrate( pBitmap );
    std::vector<Star> planarStars;
    for ( const auto& starVector : pRegistrator->GetStars() )
    {
        planarStars.insert( planarStars.end(), starVector.begin(), starVector.end() );
    }

    if ( !aligner_ )
    {
        aligner_ = std::make_unique<FastAligner>( planarStars );        
        
    }
    else
    {
        aligner_->Align( planarStars, true );
        const auto transform = aligner_->GetTransform();
        pBitmap = AffineTransform::ApplyTransform( pBitmap, {.transform = transform} );
    }

    switch ( _pixelFormat )
    {
        case PixelFormat::Gray8:
            return AddBitmapInternal<PixelFormat::Gray8>( std::static_pointer_cast< Bitmap<PixelFormat::Gray8> >(pBitmap), _means, _devs, _counts );
        case PixelFormat::Gray16:
            return AddBitmapInternal<PixelFormat::Gray16>( std::static_pointer_cast< Bitmap<PixelFormat::Gray16> >(pBitmap), _means, _devs, _counts );
        case PixelFormat::RGB24:
            return AddBitmapInternal<PixelFormat::RGB24>( std::static_pointer_cast< Bitmap<PixelFormat::RGB24> >(pBitmap), _means, _devs, _counts );
        case PixelFormat::RGB48:
            return AddBitmapInternal<PixelFormat::RGB48>( std::static_pointer_cast< Bitmap<PixelFormat::RGB48> >(pBitmap), _means, _devs, _counts );
        default:
            throw std::invalid_argument( "unsupported pixel format" );
    }
}

std::shared_ptr<IBitmap> SimpleStacker::GetResult()
{
    switch ( _pixelFormat )
    {
    case PixelFormat::Gray8:
        return GetResultInternal<PixelFormat::Gray8>( _width, _height, _means );
    case PixelFormat::Gray16:
        return GetResultInternal<PixelFormat::Gray16>( _width, _height, _means );
    case PixelFormat::RGB24:
        return GetResultInternal<PixelFormat::RGB24>( _width, _height, _means );
    case PixelFormat::RGB48:
        return GetResultInternal<PixelFormat::RGB48>( _width, _height, _means );
    default:
        throw std::invalid_argument( "unsupported pixel format" );
    }
}

BaseStacker::BaseStacker( const ImageParams& imageParams, StackMode stackMode )
    : IStacker( imageParams )
    , _stackMode( stackMode )
{
    if ( (_stackMode == StackMode::Light) && _pixelFormat == PixelFormat::Bayer16 )
        _pixelFormat = PixelFormat::RGB48;

    _gridWidth = _width / cGridPixelSize + ((_width % cGridPixelSize) ? 1 : 0);
    _gridHeight = _height / cGridPixelSize + ((_height % cGridPixelSize) ? 1 : 0);
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
        Triangle targetTriangle{ PointD {d.coords[2 * d.triangles[i]], d.coords[2 * d.triangles[i] + 1]}, PointD {d.coords[2 * d.triangles[i + 1]], d.coords[2 * d.triangles[i + 1] + 1]}, PointD {d.coords[2 * d.triangles[i + 2]], d.coords[2 * d.triangles[i + 2] + 1]} };
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

/*std::shared_ptr<IBitmap> BaseStacker::Stack()
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

    return GetResult();
}*/

void BaseStacker::AddBitmap(Pipeline pipeline)
{
    Log( pipeline.GetFileName() + " in process" );
    if ( ( _stackMode == StackMode::Light ) && pipeline.GetFinalParams()->GetPixelFormat() == PixelFormat::Bayer16 )
        pipeline.AddTransform<DebayerTransform>( pipeline.GetCameraSettings() );


    IStacker::AddBitmap( pipeline );

    auto pBitmap = pipeline.RunAndGetBitmap();
    if ( !_pCameraSettings )
        _pCameraSettings = pBitmap->GetCameraSettings();

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
    auto pRes = CallGeneratingResultHelper();
    if ( ( _stackMode == StackMode::LightNoAlign || _stackMode == StackMode::StarTrails ) && pRes->GetPixelFormat() == PixelFormat::Bayer16 )
        pRes = DebayerTransform::Debayer( pRes, _pCameraSettings );

    return pRes;
}

IBitmapPtr IStacker::ProcessBitmap( IBitmapPtr )
{
    return nullptr;
}

IBitmapPtr BaseStacker::ProcessBitmap(IBitmapPtr)
{
    return nullptr;
}

ACMB_NAMESPACE_END
