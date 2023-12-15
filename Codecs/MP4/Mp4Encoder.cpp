#include "Mp4Encoder.h"
#include "./../../Core/bitmap.h"
#include "./../../Tools/SystemTools.h"

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
//#include "x264.h"
#include "x265.h"

#include <filesystem>

ACMB_NAMESPACE_BEGIN

struct Mp4EncoderParams
{
    Mp4EncoderParams()
    : pParam( x265_param_alloc(), x265_param_free )
    , pPic( x265_picture_alloc(), x265_picture_free )
    , pPic_out( x265_picture_alloc(), x265_picture_free )
    , pEncoder( nullptr, x265_encoder_close )
    , pNal( new x265_nal )
    {

    }

    ~Mp4EncoderParams()
    {
        delete pNal;
    }

    std::unique_ptr<x265_param, decltype(&x265_param_free)> pParam;
    std::unique_ptr<x265_picture, decltype(&x265_picture_free)> pPic;
    std::unique_ptr<x265_picture, decltype(&x265_picture_free)>pPic_out;
    std::unique_ptr<x265_encoder, decltype(&x265_encoder_close)> pEncoder;
    x265_nal* pNal;

    uint32_t iNal = 0;
    uint32_t iFrame = 0;
};

Mp4Encoder::Mp4Encoder( H265Preset preset, H265Tune tune, H265Profile profile )
{
    _params = std::make_shared<Mp4EncoderParams>();

    if ( x265_param_default_preset( _params->pParam.get(), x265_preset_names[int( preset )], x265_tune_names[int( tune )] ) < 0 )
    {
        delete _params->pNal;
        throw std::runtime_error( "unable to create x265 encoder" );
    }

    _params->pParam->bRepeatHeaders = true;
    _params->pParam->bLossless = true;
    _params->pParam->internalCsp = X265_CSP_I420;
    //_params->param.i_bitdepth = 8;
    /*//_params->param.i_keyint_max = 1;
    _params->param.b_intra_refresh = 1;

    //Rate control:
    _params->param.rc.i_rc_method = X264_RC_CRF;
    _params->param.rc.f_rf_constant = 0;
    _params->param.rc.f_rf_constant_max = 35;

    _params->param.b_vfr_input = 0;
    _params->param.b_repeat_headers = 1;
    _params->param.b_annexb = 1;*/
    
    
}

void Mp4Encoder::Attach( const std::string& fileName )
{
#ifdef _WIN32
    fopen_s( &_f, fileName.c_str(), "w" );
#else
    _f = fopen( fileName.c_str(), "w" );
#endif
    if ( !_f )
        throw std::runtime_error( "unable to open file" );
}

void Mp4Encoder::Attach( std::shared_ptr<std::ostream> pStream )
{
    if ( !pStream )
        throw std::invalid_argument( "pStream" );

    _pStream = pStream;
    RandomStringGenerator rsg;
    Attach( std::filesystem::temp_directory_path().string() + "/tmp_" + rsg( 16 ) + ".tif" );
}

void Mp4Encoder::Detach()
{
    if ( _params->pEncoder )
    {
        while ( x265_encoder_encode( _params->pEncoder.get(), &_params->pNal, &_params->iNal, nullptr, _params->pPic_out.get() ) > 0 )
        {
            if ( !fwrite( _params->pNal->payload, _params->pNal->sizeBytes, 1, _f ) )
                break;
        }
    }

    _params->pPic.reset();
    _params->pPic_out.reset();
    _params->pEncoder.reset();

    _i420Channels[0].clear();
    _i420Channels[1].clear();
    _i420Channels[2].clear();

    if ( _f )
    {
        fclose( _f );
        _f = nullptr;
    }

    _pStream.reset();
}

void Mp4Encoder::WriteBitmap( std::shared_ptr<IBitmap> pBitmap )
{
    if ( !pBitmap )
        throw std::invalid_argument( "pBitmap is null" );

    if ( pBitmap->GetPixelFormat() != PixelFormat::RGB24 )
        throw std::invalid_argument( "unsupported pixel format" );
    


    if ( !_params->pEncoder )
    {
        _params->pParam->sourceWidth = pBitmap->GetWidth();
        _params->pParam->sourceHeight = pBitmap->GetHeight();

        const auto width = _params->pParam->sourceWidth;
        const auto height = _params->pParam->sourceHeight;
        
        x265_picture_init( _params->pParam.get(), _params->pPic.get() );
        _i420Channels[0].resize( width * height );
        _i420Channels[1].resize( width * height / 4 );
        _i420Channels[2].resize( width * height / 4 );

        _params->pPic->planes[0] = _i420Channels[0].data();
        _params->pPic->planes[1] = _i420Channels[1].data();
        _params->pPic->planes[2] = _i420Channels[2].data();

        _params->pPic->stride[0] = width;
        _params->pPic->stride[1] = width / 2;
        _params->pPic->stride[2] = width / 2;
        _params->pPic->width = width;
        _params->pPic->height = height;

        _params->pEncoder = std::unique_ptr<x265_encoder, decltype(&x265_encoder_close)>( x265_encoder_open( _params->pParam.get() ), x265_encoder_close );
        if ( !_params->pEncoder )
            throw std::runtime_error( "unable to create x265 encoder" );
    }
    
    const auto width = _params->pParam->sourceWidth;
    const auto height = _params->pParam->sourceHeight;

    if ( width != pBitmap->GetWidth() || height != pBitmap->GetHeight() )
        throw std::runtime_error( "bitmap size mismatch" );

    constexpr int channelCount = 3;
    tbb::parallel_for( tbb::blocked_range<int>( 0, height ), [&] ( const tbb::blocked_range<int>& range )
    {
        for ( int line = range.begin(); line < range.end(); ++line )
            //for ( int line = 0; line < height; ++line )
        {
            auto pScanline = ( uint8_t* ) pBitmap->GetPlanarScanline( line );
            int lPos = line * width;
            int uPos = line * width / 4;
            int vPos = line * width / 4;

            if ( !(line % 2) )
            {
                for ( size_t x = 0; x < width; x += 2 )
                {
                    uint8_t r = pScanline[x * channelCount];
                    uint8_t g = pScanline[x * channelCount + 1];
                    uint8_t b = pScanline[x * channelCount + 2];

                    (( uint8_t* ) _params->pPic->planes[0])[lPos++] = ((66 * r + 129 * g + 25 * b) >> 8) + 16;
                    (( uint8_t* ) _params->pPic->planes[1])[uPos++] = ((-38 * r + -74 * g + 112 * b) >> 8) + 128;
                    (( uint8_t* ) _params->pPic->planes[2])[vPos++] = ((112 * r + -94 * g + -18 * b) >> 8) + 128;

                    r = pScanline[(x + 1) * channelCount];
                    g = pScanline[(x + 1) * channelCount + 1];
                    b = pScanline[(x + 1) * channelCount + 2];

                    (( uint8_t* ) _params->pPic->planes[0])[lPos++] = ((66 * r + 129 * g + 25 * b) >> 8) + 16;
                }
            }
            else
            {
                for ( size_t x = 0; x < width; x += 1 )
                {
                    uint8_t r = pScanline[x * channelCount];
                    uint8_t g = pScanline[x * channelCount + 1];
                    uint8_t b = pScanline[x * channelCount + 2];
                    (( uint8_t* ) _params->pPic->planes[0])[lPos++] = ((66 * r + 129 * g + 25 * b) >> 8) + 16;
                }
            }
        }
    } );



            /*const int lineOffset = line * _params->pic.img.i_stride[0];
            for ( int x = 0; x < _params->param.i_width; ++x )
            {
                const int pixelOffset = lineOffset + j * channelCount;
                _params->pic.img.plane[0][pixelOffset] = pScanline[j * channelCount];
                _params->pic.img.plane[0][pixelOffset + 1] = pScanline[j * channelCount + 1];
                _params->pic.img.plane[0][pixelOffset + 2] = pScanline[j * channelCount + 2];
            }
        }
    } );*/
    const auto res = x265_encoder_encode( _params->pEncoder.get(), &_params->pNal, &_params->iNal, _params->pPic.get(), _params->pPic_out.get());
    if ( res < 0 )
    {
        throw std::runtime_error( "unable to encode x264 frame" );
    }
    else if ( res > 0 )
    {
        if ( !fwrite( _params->pNal->payload, _params->pNal->sizeBytes, 1, _f ) )
            throw std::runtime_error( "unable to encode x264 frame" );
    }
    /*

    /*auto pLumaBitmap = IBitmap::Create(width, height, PixelFormat::Gray8);
    std::copy( _params->pic.img.plane[0], _params->pic.img.plane[0] + width * height, pLumaBitmap->GetPlanarScanline( 0 ) );
    IBitmap::Save( pLumaBitmap, "F:\\Projects\\AstroCombine\\Tests\\Patterns\\Mp4Encoder\\luma.ppm" );

    auto pChromaUBitmap = IBitmap::Create( width / 2, height / 2, PixelFormat::Gray8 );
    std::copy( _params->pic.img.plane[1], _params->pic.img.plane[1] + width * height / 4, pChromaUBitmap->GetPlanarScanline( 0 ) );
    IBitmap::Save( pChromaUBitmap, "F:\\Projects\\AstroCombine\\Tests\\Patterns\\Mp4Encoder\\chroma_u.ppm" );

    auto pChromaVBitmap = IBitmap::Create( width / 2, height / 2, PixelFormat::Gray8 );
    std::copy( _params->pic.img.plane[2], _params->pic.img.plane[2] + width * height / 4, pChromaVBitmap->GetPlanarScanline( 0 ) );
    IBitmap::Save( pChromaVBitmap, "F:\\Projects\\AstroCombine\\Tests\\Patterns\\Mp4Encoder\\chroma_v.ppm" );*/

    /*_params->pic.i_pts = _params->i_frame++;
    auto i_frame_size = x264_encoder_encode( _params->h, &_params->nal, &_params->i_nal, &_params->pic, &_params->pic_out );
    if ( i_frame_size < 0 )
    {
        throw std::runtime_error( "unable to encode x264 frame" );
    }
    else if ( i_frame_size )
    {
        if ( !fwrite( _params->nal->p_payload, i_frame_size, 1, _f ) )
            throw std::runtime_error( "unable to encode x264 frame" );
    }*/
}

void Mp4Encoder::SetFrameRate( uint32_t rate )
{
    if ( _params )
    {
        _params->pParam->fpsNum = rate;
        _params->pParam->fpsDenom = 1;
    }
}

uint32_t Mp4Encoder::GetFrameRate() const
{
    return _params ? _params->pParam->fpsNum : 0;
}

std::set<std::string> Mp4Encoder::GetExtensions()
{
    return { ".mp4", ".h264" };
}

ACMB_NAMESPACE_END