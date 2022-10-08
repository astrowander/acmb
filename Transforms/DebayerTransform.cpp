#include "DebayerTransform.h"
#include <libraw/libraw.h>
#define BLACK_LEVEL 2048

ACMB_NAMESPACE_BEGIN

DebayerTransform::DebayerTransform( IBitmapPtr pSrcBitmap, std::shared_ptr<CameraSettings> pCameraSettings )
: BaseTransform( pSrcBitmap )
, _pCameraSettings( pCameraSettings )
{
    if ( pSrcBitmap->GetPixelFormat() != PixelFormat::Gray16 )
        throw std::invalid_argument( "only extended grayscale images can be debayered" );
}

std::shared_ptr<DebayerTransform> DebayerTransform::Create( IBitmapPtr pSrcBitmap, std::shared_ptr<CameraSettings> pCameraSettings )
{
    return std::make_shared<DebayerTransform>( pSrcBitmap, pCameraSettings );
}

/// Creates instance with source pixel format and camera settings. Source bitmap must be set later

std::shared_ptr<DebayerTransform> DebayerTransform::Create( PixelFormat pixelFormat, std::shared_ptr<CameraSettings> pCameraSettings )
{
    return std::make_shared<DebayerTransform>( nullptr, pCameraSettings );
}

void DebayerTransform::Run()
{
    _pDstBitmap = IBitmap::Create( _pSrcBitmap->GetWidth(), _pSrcBitmap->GetHeight(), ConstructPixelFormat( BitsPerChannel( _pSrcBitmap->GetPixelFormat() ), 3 ) );

    LibRaw libRaw;
    libRaw.imgdata.params.output_bps = BitsPerChannel( _pSrcBitmap->GetPixelFormat() );
    libRaw.imgdata.params.no_interpolation = 0;
    libRaw.imgdata.params.fbdd_noiserd = 0;
    libRaw.imgdata.params.med_passes = 0;
    libRaw.imgdata.params.no_auto_bright = 1;
    libRaw.imgdata.params.half_size = false;
    libRaw.imgdata.params.user_qual = 0;

    if ( libRaw.open_bayer( ( uint8_t* ) _pSrcBitmap->GetPlanarScanline( 0 ),
                            _pSrcBitmap->GetByteSize(), _pSrcBitmap->GetWidth(), _pSrcBitmap->GetHeight(), 0, 0, 0, 0, 0, LIBRAW_OPENBAYER_RGGB, 0, 0, BLACK_LEVEL ) != LIBRAW_SUCCESS )
    {
        throw std::runtime_error( "unable to debayer" );
    }

    if ( libRaw.unpack() != LIBRAW_SUCCESS )
        throw std::runtime_error( "unable to unpack " );

    if ( libRaw.dcraw_process() != LIBRAW_SUCCESS )
        throw std::runtime_error( "unable to process " );

    libRaw.imgdata.sizes.flip = 0;
    int ret = 0;
    libraw_processed_image_t* image = libRaw.dcraw_make_mem_image( &ret );
    if ( ret != LIBRAW_SUCCESS )
    {
        libRaw.dcraw_clear_mem( image );
        throw std::runtime_error( "processing error" );
    }
    memcpy( _pDstBitmap->GetPlanarScanline( 0 ), image->data, image->data_size );

    libRaw.dcraw_clear_mem( image );
}

void DebayerTransform::CalcParams( std::shared_ptr<ImageParams> pParams )
{
    if ( !pParams )
        throw std::invalid_argument( "pParams is null" );

    _width = pParams->GetWidth();
    _height = pParams->GetHeight();
    _pixelFormat = ConstructPixelFormat( BitsPerChannel( pParams->GetPixelFormat() ) , 3 );
}

ACMB_NAMESPACE_END

/// Creates instance with source bitmap and camera settings


