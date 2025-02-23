#include "H265Encoder.h"
#include "./../../Core/bitmap.h"
#include "./../../Tools/BitmapTools.h"

#include "x265.h"

ACMB_NAMESPACE_BEGIN

struct H265EncoderParams
{
    H265EncoderParams()
    : pParam( x265_param_alloc(), x265_param_free )
    , pPic( x265_picture_alloc(), x265_picture_free )
    , pPicOut( x265_picture_alloc(), x265_picture_free )
    , pEncoder( nullptr, x265_encoder_close )
    {

    }

    std::unique_ptr<x265_param, decltype(&x265_param_free)> pParam;
    std::unique_ptr<x265_picture, decltype(&x265_picture_free)> pPic;
    std::unique_ptr<x265_picture, decltype(&x265_picture_free)> pPicOut;
    std::unique_ptr<x265_encoder, decltype(&x265_encoder_close)> pEncoder;
    x265_nal* pNal = nullptr;

    uint32_t iNal = 0;
    uint32_t iFrame = 0;
};

H265Encoder::H265Encoder( Preset preset, Tune tune, Profile profile )
{
    _params = std::make_shared<H265EncoderParams>();

    if ( x265_param_default_preset( _params->pParam.get(), x265_preset_names[int( preset )], x265_tune_names[int( tune )] ) < 0 )
    {
        throw std::runtime_error( "unable to create x265 encoder" );
    }

    _params->pParam->bRepeatHeaders = true;
    _params->pParam->internalCsp = X265_CSP_I420;

    if ( x265_param_apply_profile( _params->pParam.get(), x265_profile_names[int( profile )] ) < 0 )
    {
        throw std::runtime_error( "unable to apply profile" );
    }

    _params->pParam->rc.rfConstant = 19;
    _params->pParam->psyRd = 2.0f;
    _params->pParam->psyRdoq = 2.0f;
    _params->pParam->rc.aqMode = 1;
    _params->pParam->rc.aqStrength = 1.5;
}

void H265Encoder::Detach()
{
    if ( _params->pEncoder )
    {
        while ( x265_encoder_encode( _params->pEncoder.get(), &_params->pNal, &_params->iNal, nullptr, _params->pPicOut.get() ) > 0 )
        {
            for ( uint32_t i = 0; i < _params->iNal; ++i )
                _pStream->write( (const char *)_params->pNal[i].payload, _params->pNal[i].sizeBytes );
        }
    }

    _params->pPic.reset();
    _params->pPicOut.reset();
    _params->pEncoder.reset();

    _pStream.reset();
    _yuv.clear();
}

void H265Encoder::WriteBitmap( std::shared_ptr<IBitmap> pBitmap )
{
    if ( !pBitmap )
        throw std::invalid_argument( "pBitmap is null" );

    if ( pBitmap->GetPixelFormat() != PixelFormat::YUV24 )
        throw std::invalid_argument( "H265Encoder: unsupported pixel format" );

    if ( _width == 0 && _height == 0 )
    {
        _width = pBitmap->GetWidth();
        _height = pBitmap->GetHeight();
        _params->pParam->sourceWidth = _width;
        _params->pParam->sourceHeight = _height;
        _params->pParam->fpsNum = _frameRate;
        _params->pParam->fpsDenom = 1;

        _yuv.resize( _width * _height * 3 / 2 );

        x265_picture_init( _params->pParam.get(), _params->pPic.get() );
        _params->pPic->planes[0] = _yuv.data();
        _params->pPic->planes[1] = _yuv.data() + _width * _height;
        _params->pPic->planes[2] = _yuv.data() + _width * _height * 5 / 4;

        _params->pPic->stride[0] = _width;
        _params->pPic->stride[1] = _width / 2;
        _params->pPic->stride[2] = _width / 2;
        _params->pPic->width = _width;
        _params->pPic->height = _height;

        _params->pEncoder = std::unique_ptr<x265_encoder, decltype(&x265_encoder_close)>( x265_encoder_open( _params->pParam.get() ), x265_encoder_close );
        if ( !_params->pEncoder )
            throw std::runtime_error( "unable to create x265 encoder" );
    }

    if ( pBitmap->GetWidth() != _width || pBitmap->GetHeight() != _height )
        throw std::runtime_error( "bitmap size mismatch" );

    PlanarDataFromYUVBitmap( std::static_pointer_cast< Bitmap<PixelFormat::YUV24> >(pBitmap), _yuv );
    _params->pPic->pts = _params->iFrame++;
    const auto res = x265_encoder_encode( _params->pEncoder.get(), &_params->pNal, &_params->iNal, _params->pPic.get(), _params->pPicOut.get() );
    if ( res < 0 )
    {
        throw std::runtime_error( "unable to encode x264 frame" );
    }
    else if ( res > 0 )
    {
        for ( uint32_t i = 0; i < _params->iNal; ++i )
        {
            _pStream->write( ( const char* ) _params->pNal[i].payload, _params->pNal[i].sizeBytes );
        }
    }
}

std::set<std::string> H265Encoder::GetExtensions()
{
    return { ".h265" };
}


ACMB_NAMESPACE_END