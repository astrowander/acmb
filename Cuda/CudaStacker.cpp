#include "CudaStacker.h"
#include "AddBitmap.h"
#include "AddBitmapWithAlignment.h"
#include "GenerateResult.h"

#include "./../Core/log.h"
#include "./../Registrator/registrator.h"
ACMB_CUDA_NAMESPACE_BEGIN

void CallAddBitmapHelper( IBitmapPtr pBitmap, float* pMeans, float* pDevs, uint16_t * pCounts, size_t size )
{
    switch ( pBitmap->GetPixelFormat() )
    {
        case PixelFormat::Gray8:
        {
            DynamicArrayU8 pPixels( std::static_pointer_cast< Bitmap<PixelFormat::Gray8> >( pBitmap )->GetData() );
            AddBitmapHelper( pPixels.data(), pMeans, pDevs, pCounts, size);
            break;
        }
        case PixelFormat::Gray16:
        {
            DynamicArrayU16 pPixels( std::static_pointer_cast< Bitmap<PixelFormat::Gray16> >( pBitmap )->GetData() );
            AddBitmapHelper( pPixels.data(), pMeans, pDevs, pCounts, size );
            break;
        }
        case PixelFormat::RGB24:
        {
            DynamicArrayU8 pPixels( std::static_pointer_cast< Bitmap<PixelFormat::RGB24> >( pBitmap )->GetData() );
            AddBitmapHelper( pPixels.data(), pMeans, pDevs, pCounts, size );
            break;
        }
        case PixelFormat::RGB48:
        {
            DynamicArrayU16 pPixels( std::static_pointer_cast< Bitmap<PixelFormat::RGB48> >( pBitmap )->GetData() );            
            AddBitmapHelper( pPixels.data(), pMeans, pDevs, pCounts, size);
            break;
        }
        case PixelFormat::Bayer16:
        {
            DynamicArrayU16 pPixels( std::static_pointer_cast< Bitmap<PixelFormat::Bayer16> >( pBitmap )->GetData() );
            AddBitmapHelper( pPixels.data(), pMeans, pDevs, pCounts, size );
            break;
        }
        default:
            throw std::runtime_error( "pixel format should be known" );
    }
}

IBitmapPtr CallGeneratingResultHelper( float* pMeans, uint32_t width, uint32_t height, PixelFormat pixelFormat )
{
    switch ( pixelFormat )
    {
        case PixelFormat::RGB24:
            return GeneratingResultHelper<PixelFormat::RGB24>( pMeans, width, height );
        case acmb::PixelFormat::RGB48:
            return GeneratingResultHelper<PixelFormat::RGB48>( pMeans, width, height );
        case acmb::PixelFormat::Gray8:
            return GeneratingResultHelper<PixelFormat::Gray8>( pMeans, width, height );
        case acmb::PixelFormat::Gray16:
            return GeneratingResultHelper<PixelFormat::Gray16>( pMeans, width, height );
        case acmb::PixelFormat::Bayer16:
            return GeneratingResultHelper<PixelFormat::Bayer16>( pMeans, width, height );
        default:
            throw std::runtime_error( "pixel format should be known" );
    }
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
            _cudaBitmap =  std::move( DynamicArrayU8(size) );
            break;
        case PixelFormat::Gray16:
        case PixelFormat::Bayer16:
        case PixelFormat::RGB48:
            _cudaBitmap = std::move( DynamicArrayU16( size ) );
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

    const size_t size = _width * _height * ChannelCount( pRefBitmap->GetPixelFormat() );
    _means.resize( size );
    _devs.resize( size );
    _counts.resize( size );

    CallAddBitmapHelper( pRefBitmap, _means.data(), _devs.data(), _counts.data(), size );
   // if ( cudaDeviceSynchronize() != cudaSuccess )
     //   throw std::runtime_error( "error in CUDA kernel occured" );

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
            CallAddBitmapHelper( pTargetBitmap, _means.data(), _devs.data(), _counts.data(), size );
           // if ( cudaDeviceSynchronize() != cudaSuccess )
               // throw std::runtime_error( "error in CUDA kernel occured" );
            Log( _stackingData[i].pipeline.GetFileName() + " is stacked" );
            continue;
        }

        CalculateAligningGrid( i );

        Log( _stackingData[i].pipeline.GetFileName() + " grid is calculated" );

        AddBitmapWithAlignmentHelper( pTargetBitmap, _means.data(), _devs.data(), _counts.data(), _grid, gridSize, _gridWidth );
       // if ( cudaDeviceSynchronize() != cudaSuccess )
         //   throw std::runtime_error( "error in CUDA kernel occured" );

        Log( _stackingData[i].pipeline.GetFileName() + " is stacked" );
    }

    const auto res = CallGeneratingResultHelper( _means.data(), _width, _height, _pixelFormat );
   // if ( cudaDeviceSynchronize() != cudaSuccess )
     //   throw std::runtime_error( "error in CUDA kernel occured" );

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

    //auto errCode = cudaDeviceSynchronize();
    //if ( errCode != cudaSuccess )
     //   throw std::runtime_error( "error in CUDA kernel occured" );
    const size_t size = _width * _height * ChannelCount( _pixelFormat );
    CallAddBitmapHelper( pRefBitmap, _means.data(), _devs.data(), _counts.data(), size );
   // errCode = cudaDeviceSynchronize();
   // if (  errCode != cudaSuccess )
    //    throw std::runtime_error( "error in CUDA kernel occured" );

    Log( _stackingData[0].pipeline.GetFileName() + " is stacked" );

    for ( uint32_t i = 1; i < _stackingData.size(); ++i )
    {
        Log( _stackingData[i].pipeline.GetFileName() + " in process" );
        auto pTargetBitmap = _stackingData[i].pipeline.RunAndGetBitmap();
        Log( _stackingData[i].pipeline.GetFileName() + " is read" );

       // if ( pRefBitmap->GetPixelFormat() != pTargetBitmap->GetPixelFormat() )
         //   throw std::runtime_error( "bitmaps in stack should have the same pixel format" );

        pRegistrator->Registrate( pTargetBitmap );
        _stackingData[i].stars = pRegistrator->GetStars();

        if ( _stackMode != StackMode::Light )
        {
            CallAddBitmapHelper( pTargetBitmap, _means.data(), _devs.data(), _counts.data(), size );
           // if ( cudaDeviceSynchronize() != cudaSuccess )
             //   throw std::runtime_error( "error in CUDA kernel occured" );
            Log( _stackingData[i].pipeline.GetFileName() + " is stacked" );
            continue;
        }

        CalculateAligningGrid( i );

        Log( _stackingData[i].pipeline.GetFileName() + " grid is calculated" );

        AddBitmapWithAlignmentHelper( pTargetBitmap, _means.data(), _devs.data(), _counts.data(), _grid, gridSize, _gridWidth );
       // if ( cudaDeviceSynchronize() != cudaSuccess )
       //     throw std::runtime_error( "error in CUDA kernel occured" );

        Log( _stackingData[i].pipeline.GetFileName() + " is stacked" );
    }

    const auto res = CallGeneratingResultHelper( _means.data(), _width, _height, _pixelFormat );
  //  if ( cudaDeviceSynchronize() != cudaSuccess )
   //     throw std::runtime_error( "error in CUDA kernel occured" );
    return res;
}


ACMB_CUDA_NAMESPACE_END