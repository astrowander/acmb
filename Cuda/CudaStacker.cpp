#include "CudaStacker.h"
#include "AddBitmap.h"
#include "GenerateResult.h"
#include "CudaBasic.h"
#include "./../Core/bitmap.h"

#define CUDA_SYNCHRONIZE
ACMB_CUDA_NAMESPACE_BEGIN

struct StackData
{
    DynamicArrayF _means;
    DynamicArrayF _devs;
    DynamicArrayU16 _counts;
    std::variant<DynamicArrayU8, DynamicArrayU16> _cudaBitmap;
};

Stacker::Stacker( const std::vector<Pipeline>& pipelines, StackMode stackMode )
: BaseStacker(pipelines, stackMode)
{ 
    Init();
}

Stacker::Stacker( const ImageParams& imageParams, StackMode stackMode )
: BaseStacker(imageParams, stackMode)
{
    Init();
}

void Stacker::Init()
{
    _stackData = std::make_shared<StackData>();

    const size_t size = _width * _height * ChannelCount( _pixelFormat );
    _stackData->_means.resize( size );
    _stackData->_devs.resize( size );
    _stackData->_counts.resize( size );

    switch ( _pixelFormat )
    {
        case PixelFormat::Gray8:
        case PixelFormat::RGB24:
            _stackData->_cudaBitmap =   DynamicArrayU8(size) ;
            _helper = AddBitmapWithAlignmentHelperU8();
            break;
        case PixelFormat::Gray16:
        case PixelFormat::Bayer16:
        case PixelFormat::RGB48:
            _stackData->_cudaBitmap =  DynamicArrayU16( size ) ;
            _helper = AddBitmapWithAlignmentHelperU16();
            break;
        default:
            throw std::runtime_error( "pixel format should be known" );
    }
}


void Stacker::CallAddBitmapHelper( IBitmapPtr pBitmap )
{
#ifdef CUDA_SYNCHRONIZE
    if ( cudaDeviceSynchronize() != cudaSuccess )
        throw std::runtime_error( "error in CUDA kernel occured" );
#endif
    const size_t size = _width * _height * ChannelCount( _pixelFormat );

#define TRY_ADD_BITMAP( format ) \
if (_pixelFormat == format) { \
    using DynamicArrayT = typename std::conditional_t<PixelFormatTraits<format>::bytesPerChannel == 1, DynamicArrayU8, DynamicArrayU16>;\
    auto& bitmap = std::get<DynamicArrayT>( _stackData->_cudaBitmap );\
    bitmap.fromVector( std::static_pointer_cast< Bitmap<format> >( pBitmap )->GetData() );\
    return AddBitmapHelper( bitmap.data(), _stackData->_means.data(), _stackData->_devs.data(), _stackData->_counts.data(), size ); }

    TRY_ADD_BITMAP( PixelFormat::Gray8 );
    TRY_ADD_BITMAP( PixelFormat::Gray16 );
    TRY_ADD_BITMAP( PixelFormat::RGB24 );
    TRY_ADD_BITMAP( PixelFormat::RGB48 );
    TRY_ADD_BITMAP( PixelFormat::Bayer16 );

    throw std::runtime_error( "pixel format must be known" );
#ifdef CUDA_SYNCHRONIZE
    if ( cudaDeviceSynchronize() != cudaSuccess )
        throw std::runtime_error( "error in CUDA kernel occured" );
#endif
}

void Stacker::CallAddBitmapWithAlignmentHelper( IBitmapPtr pBitmap )
{
#ifdef CUDA_SYNCHRONIZE
    if ( cudaDeviceSynchronize() != cudaSuccess )
        throw std::runtime_error( "error in CUDA kernel occured" );
#endif
#define TRY_ADD_BITMAP_WITH_ALIGNMENT( format ) \
if (_pixelFormat == format) { \
    using DynamicArrayT = typename std::conditional_t<PixelFormatTraits<format>::bytesPerChannel == 1, DynamicArrayU8, DynamicArrayU16>;\
    using HelperT = typename std::conditional_t<PixelFormatTraits<format>::bytesPerChannel == 1, AddBitmapWithAlignmentHelperU8, AddBitmapWithAlignmentHelperU16>;\
    auto& bitmap = std::get<DynamicArrayT>( _stackData->_cudaBitmap );\
    auto& helper = std::get<HelperT>( _helper );\
    bitmap.fromVector( std::static_pointer_cast< Bitmap<format> >( pBitmap )->GetData() );\
    return helper.Run( bitmap.data(), _width, _height, PixelFormatTraits<format>::channelCount, _grid, _stackData->_means.data(), _stackData->_devs.data(), _stackData->_counts.data() ); }

    TRY_ADD_BITMAP_WITH_ALIGNMENT( PixelFormat::Gray8 );
    TRY_ADD_BITMAP_WITH_ALIGNMENT( PixelFormat::Gray16 );
    TRY_ADD_BITMAP_WITH_ALIGNMENT( PixelFormat::RGB24 );
    TRY_ADD_BITMAP_WITH_ALIGNMENT( PixelFormat::RGB48 );
    TRY_ADD_BITMAP_WITH_ALIGNMENT( PixelFormat::Bayer16 );

    throw std::runtime_error( "pixel format must be known" );
#ifdef CUDA_SYNCHRONIZE
    if ( cudaDeviceSynchronize() != cudaSuccess )
        throw std::runtime_error( "error in CUDA kernel occured" );
#endif
}

IBitmapPtr Stacker::CallGeneratingResultHelper()
{
#ifdef CUDA_SYNCHRONIZE
    if ( cudaDeviceSynchronize() != cudaSuccess )
        throw std::runtime_error( "error in CUDA kernel occured" );
#endif
    const size_t size = _width * _height * ChannelCount( _pixelFormat );
#define TRY_GENERATE_RESULT( format ) \
if (_pixelFormat == format ) { \
    using DynamicArrayT = typename std::conditional_t<PixelFormatTraits<format>::bytesPerChannel == 1, DynamicArrayU8, DynamicArrayU16>;\
    auto& bitmap = std::get<DynamicArrayT>( _stackData->_cudaBitmap );\
    GeneratingResultKernel(_stackData->_means.data(), bitmap.data(), size );\
    IBitmapPtr res = IBitmap::Create( _width, _height, _pixelFormat );\
    bitmap.toVector(std::static_pointer_cast<Bitmap<format>>(res)->GetData());\
    return res; } \

    TRY_GENERATE_RESULT( PixelFormat::Gray8 );
    TRY_GENERATE_RESULT( PixelFormat::Gray16 );
    TRY_GENERATE_RESULT( PixelFormat::RGB24 );
    TRY_GENERATE_RESULT( PixelFormat::RGB48 );
    TRY_GENERATE_RESULT( PixelFormat::Bayer16 );

    throw std::runtime_error( "pixel format must be known" );
#ifdef CUDA_SYNCHRONIZE
    if ( cudaDeviceSynchronize() != cudaSuccess )
        throw std::runtime_error( "error in CUDA kernel occured" );
#endif
}

ACMB_CUDA_NAMESPACE_END
