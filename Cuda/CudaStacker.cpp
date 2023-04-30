#include "CudaStacker.h"
#include "AddBitmap.h"
#include "GenerateResult.h"

#include "./../Core/log.h"
#include "./../Registrator/registrator.h"

//#define CUDA_SYNCHRONIZE

ACMB_CUDA_NAMESPACE_BEGIN

void Stacker::CallAddBitmapHelper( IBitmapPtr pBitmap )
{
    const size_t size = _width * _height * ChannelCount( _pixelFormat );

#define TRY_ADD_BITMAP( format ) \
if (_pixelFormat == format) { \
    using DynamicArrayT = typename std::conditional_t<PixelFormatTraits<format>::bytesPerChannel == 1, DynamicArrayU8, DynamicArrayU16>;\
    auto& bitmap = std::get<DynamicArrayT>( _cudaBitmap );\
    bitmap.fromVector( std::static_pointer_cast< Bitmap<format> >( pBitmap )->GetData() );\
    return AddBitmapHelper( bitmap.data(), _means.data(), _devs.data(), _counts.data(), size ); }

    TRY_ADD_BITMAP( PixelFormat::Gray8 );
    TRY_ADD_BITMAP( PixelFormat::Gray16 );
    TRY_ADD_BITMAP( PixelFormat::RGB24 );
    TRY_ADD_BITMAP( PixelFormat::RGB48 );
    TRY_ADD_BITMAP( PixelFormat::Bayer16 );

    throw std::runtime_error( "pixel format must be known" );
}

void Stacker::CallAddBitmapWithAlignmentHelper( IBitmapPtr pBitmap, const Grid& grid )
{  
#define TRY_ADD_BITMAP_WITH_ALIGNMENT( format ) \
if (_pixelFormat == format) { \
    using DynamicArrayT = typename std::conditional_t<PixelFormatTraits<format>::bytesPerChannel == 1, DynamicArrayU8, DynamicArrayU16>;\
    using HelperT = typename std::conditional_t<PixelFormatTraits<format>::bytesPerChannel == 1, AddBitmapWithAlignmentHelperU8, AddBitmapWithAlignmentHelperU16>;\
    auto& bitmap = std::get<DynamicArrayT>( _cudaBitmap );\
    auto& helper = std::get<HelperT>( _helper );\
    bitmap.fromVector( std::static_pointer_cast< Bitmap<format> >( pBitmap )->GetData() );\
    return helper.Run( bitmap.data(), _width, _height, PixelFormatTraits<format>::channelCount, grid, _means.data(), _devs.data(), _counts.data() ); }

    TRY_ADD_BITMAP_WITH_ALIGNMENT( PixelFormat::Gray8 );
    TRY_ADD_BITMAP_WITH_ALIGNMENT( PixelFormat::Gray16 );
    TRY_ADD_BITMAP_WITH_ALIGNMENT( PixelFormat::RGB24 );
    TRY_ADD_BITMAP_WITH_ALIGNMENT( PixelFormat::RGB48 );
    TRY_ADD_BITMAP_WITH_ALIGNMENT( PixelFormat::Bayer16 );

    throw std::runtime_error( "pixel format must be known" );
}

IBitmapPtr Stacker::CallGeneratingResultHelper()
{
    const size_t size = _width * _height * ChannelCount( _pixelFormat );
#define TRY_GENERATE_RESULT( format ) \
if (_pixelFormat == format ) { \
    using DynamicArrayT = typename std::conditional_t<PixelFormatTraits<format>::bytesPerChannel == 1, DynamicArrayU8, DynamicArrayU16>;\
    auto& bitmap = std::get<DynamicArrayT>( _cudaBitmap );\
    GeneratingResultKernel(_means.data(), bitmap.data(), size );\
    IBitmapPtr res = IBitmap::Create( _width, _height, _pixelFormat );\
    bitmap.toVector(std::static_pointer_cast<Bitmap<format>>(res)->GetData());\
    return res; } \

    TRY_GENERATE_RESULT( PixelFormat::Gray8 );
    TRY_GENERATE_RESULT( PixelFormat::Gray16 );
    TRY_GENERATE_RESULT( PixelFormat::RGB24 );
    TRY_GENERATE_RESULT( PixelFormat::RGB48 );
    TRY_GENERATE_RESULT( PixelFormat::Bayer16 );

    throw std::runtime_error( "pixel format must be known" );
}

Stacker::Stacker( const std::vector<Pipeline>& pipelines, StackMode stackMode )
: BaseStacker(pipelines, stackMode)
{ 
    const size_t size = _width * _height * ChannelCount( _pixelFormat );
    _means.resize( size );
    _devs.resize( size );
    _counts.resize( size );

    switch ( _pixelFormat )
    {
        case PixelFormat::Gray8:
        case PixelFormat::RGB24:
        _cudaBitmap =   DynamicArrayU8(size) ;
            _helper = AddBitmapWithAlignmentHelperU8();
            break;
        case PixelFormat::Gray16:
        case PixelFormat::Bayer16:
        case PixelFormat::RGB48:
        _cudaBitmap =  DynamicArrayU16( size ) ;
            _helper = AddBitmapWithAlignmentHelperU16();
            break;
        default:
            throw std::runtime_error( "pixel format should be known" );
    }
}

std::shared_ptr<IBitmap> Stacker::Stack()
{
    if ( _stackingData.size() == 0 )
        return nullptr;

    Log( _stackingData[0].pipeline.GetFileName() + " in process" );

    auto pRefBitmap = _stackingData[0].pipeline.RunAndGetBitmap();

    Log( _stackingData[0].pipeline.GetFileName() + " is read" );

    if ( _stackingData.size() == 1 )
        return pRefBitmap;

    const auto& refStars = _stackingData[0].stars;

    if ( _stackMode == StackMode::Light )
    {
        _aligners.clear();

        for ( const auto& refStarVector : refStars )
            _aligners.push_back( std::make_shared<FastAligner>( refStarVector ) );
    }

    CallAddBitmapHelper( pRefBitmap );
#ifdef CUDA_SYNCHRONIZE
    if ( cudaDeviceSynchronize() != cudaSuccess )
        throw std::runtime_error( "error in CUDA kernel occured" );
#endif

    Log( _stackingData[0].pipeline.GetFileName() + " is stacked" );

    for ( uint32_t i = 1; i < _stackingData.size(); ++i )
    {
        Log( _stackingData[i].pipeline.GetFileName() + " in process" );
        auto pTargetBitmap = _stackingData[i].pipeline.RunAndGetBitmap();
        Log( _stackingData[i].pipeline.GetFileName() + " is read" );

        if ( pRefBitmap->GetPixelFormat() != pTargetBitmap->GetPixelFormat() )
            throw std::runtime_error( "bitmaps in stack should have the same pixel format" );

        if ( _stackMode != StackMode::Light )
        {
            CallAddBitmapHelper( pTargetBitmap );
#ifdef CUDA_SYNCHRONIZE
            if ( cudaDeviceSynchronize() != cudaSuccess )
                throw std::runtime_error( "error in CUDA kernel occured" );
#endif
            Log( _stackingData[i].pipeline.GetFileName() + " is stacked" );
            continue;
        }

        CalculateAligningGrid( i );

        Log( _stackingData[i].pipeline.GetFileName() + " grid is calculated" );

        CallAddBitmapWithAlignmentHelper( pTargetBitmap, _grid );
#ifdef CUDA_SYNCHRONIZE
        if ( cudaDeviceSynchronize() != cudaSuccess )
            throw std::runtime_error( "error in CUDA kernel occured" );
#endif

        Log( _stackingData[i].pipeline.GetFileName() + " is stacked" );
    }

    const auto res = CallGeneratingResultHelper();
#ifdef CUDA_SYNCHRONIZE
    if ( cudaDeviceSynchronize() != cudaSuccess )
        throw std::runtime_error( "error in CUDA kernel occured" );
#endif

    return res;
}

std::shared_ptr<IBitmap> Stacker::RegistrateAndStack()
{
    if ( _stackingData.size() == 0 )
        return nullptr;

    auto pRefBitmap = _stackingData[0].pipeline.RunAndGetBitmap();

    if ( _stackingData.size() == 1 )
        return pRefBitmap;

    auto pRegistrator = std::make_unique<Registrator>( _threshold, _minStarSize, _maxStarSize );

    pRegistrator->Registrate( pRefBitmap );
    _stackingData[0].stars = pRegistrator->GetStars();

    const auto& refStars = _stackingData[0].stars;

    _aligners.clear();
    for ( const auto& refStarVector : refStars )
        _aligners.push_back( std::make_shared<FastAligner>( refStarVector ) );    

#ifdef CUDA_SYNCHRONIZE
    if ( cudaDeviceSynchronize() != cudaSuccess )
        throw std::runtime_error( "error in CUDA kernel occured" );
#endif
    CallAddBitmapHelper( pRefBitmap );
#ifdef CUDA_SYNCHRONIZE
    if ( cudaDeviceSynchronize() != cudaSuccess )
        throw std::runtime_error( "error in CUDA kernel occured" );
#endif

    Log( _stackingData[0].pipeline.GetFileName() + " is stacked" );

    for ( uint32_t i = 1; i < _stackingData.size(); ++i )
    {
        Log( _stackingData[i].pipeline.GetFileName() + " in process" );
        auto pTargetBitmap = _stackingData[i].pipeline.RunAndGetBitmap();
        Log( _stackingData[i].pipeline.GetFileName() + " is read" );

        if ( pRefBitmap->GetPixelFormat() != pTargetBitmap->GetPixelFormat() )
            throw std::runtime_error( "bitmaps in stack should have the same pixel format" );

        pRegistrator->Registrate( pTargetBitmap );
        _stackingData[i].stars = pRegistrator->GetStars();

        if ( _stackMode != StackMode::Light )
        {
            CallAddBitmapHelper( pTargetBitmap );
#ifdef CUDA_SYNCHRONIZE
            if ( cudaDeviceSynchronize() != cudaSuccess )
                throw std::runtime_error( "error in CUDA kernel occured" );
#endif
            Log( _stackingData[i].pipeline.GetFileName() + " is stacked" );
            continue;
        }

        CalculateAligningGrid( i );

        Log( _stackingData[i].pipeline.GetFileName() + " grid is calculated" );

        CallAddBitmapWithAlignmentHelper( pTargetBitmap, _grid );
#ifdef CUDA_SYNCHRONIZE
        if ( cudaDeviceSynchronize() != cudaSuccess )
            throw std::runtime_error( "error in CUDA kernel occured" );
#endif

        Log( _stackingData[i].pipeline.GetFileName() + " is stacked" );
    }

    const auto res = CallGeneratingResultHelper();
#ifdef CUDA_SYNCHRONIZE
    if ( cudaDeviceSynchronize() != cudaSuccess )
        throw std::runtime_error( "error in CUDA kernel occured" );
#endif
    return res;
}


ACMB_CUDA_NAMESPACE_END
