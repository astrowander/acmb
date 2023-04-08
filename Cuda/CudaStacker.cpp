#include "CudaStacker.h"
#include "AddBitmap.h"

#include "./../Core/log.h"
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
            AddBitmapHelper( pPixels.data(), pMeans, pDevs, pCounts, size );
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

Stacker::Stacker( const std::vector<Pipeline>& pipelines, StackMode stackMode )
: BaseStacker(pipelines, stackMode)
{ }

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
            Log( _stackingData[i].pipeline.GetFileName() + " is stacked" );
            continue;
        }


        CalculateAligningGrid( i );

        Log( _stackingData[i].pipeline.GetFileName() + " grid is calculated" );

      //  CALL_HELPER( AddingBitmapWithAlignmentHelper, pTargetBitmap );

        Log( _stackingData[i].pipeline.GetFileName() + " is stacked" );
    }

    return pRefBitmap;
}

std::shared_ptr<IBitmap> Stacker::RegistrateAndStack()
{
    return nullptr;
}


ACMB_CUDA_NAMESPACE_END