#include "DebayerTransform.h"
#include "../Core/camerasettings.h"
#include <libraw/libraw.h>

ACMB_NAMESPACE_BEGIN

DebayerTransform::DebayerTransform( IBitmapPtr pSrcBitmap, std::shared_ptr<CameraSettings> pCameraSettings )
: BaseTransform( pSrcBitmap )
, _pCameraSettings( pCameraSettings )
{
    if ( !pCameraSettings )
        throw std::invalid_argument( "pCameraSettings is null" );

    if ( pSrcBitmap && pSrcBitmap->GetPixelFormat() != PixelFormat::Gray16 )
        throw std::invalid_argument( "only extended grayscale images can be debayered" );
}

std::shared_ptr<DebayerTransform> DebayerTransform::Create( IBitmapPtr pSrcBitmap, std::shared_ptr<CameraSettings> pCameraSettings )
{
    if ( !pSrcBitmap )
        throw std::invalid_argument( "pCameraSettings is null" );

    if ( !pCameraSettings )
        throw std::invalid_argument( "pCameraSettings is null" );

    return std::make_shared<DebayerTransform>( pSrcBitmap, pCameraSettings );
}

std::shared_ptr<DebayerTransform> DebayerTransform::Create( PixelFormat pixelFormat, std::shared_ptr<CameraSettings> pCameraSettings )
{
    if ( !pCameraSettings )
        throw std::invalid_argument( "pCameraSettings is null" );

    return std::make_shared<DebayerTransform>( nullptr, pCameraSettings );
}

IBitmapPtr DebayerTransform::Debayer( IBitmapPtr pSrcBitmap, std::shared_ptr<CameraSettings> pCameraSettings )
{
    DebayerTransform db( pSrcBitmap, pCameraSettings );
    return db.RunAndGetBitmap();
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
    const auto black = _pCameraSettings->blackLevel;
    if ( libRaw.open_bayer( ( uint8_t* ) _pSrcBitmap->GetPlanarScanline( 0 ),
                            _pSrcBitmap->GetByteSize(), _pSrcBitmap->GetWidth(), _pSrcBitmap->GetHeight(), 0, 0, 0, 0, 0, LIBRAW_OPENBAYER_RGGB, 0, 0, black ) != LIBRAW_SUCCESS )
    {
        throw std::runtime_error( "unable to debayer" );
    }
    libRaw.imgdata.color.maximum = _pCameraSettings->maxChannel;
    for ( int i = 0; i < 4; ++i )
    {
        libRaw.imgdata.color.pre_mul[i] = _pCameraSettings->channelPremultipiers[i];
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


