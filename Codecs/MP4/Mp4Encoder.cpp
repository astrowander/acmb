#include "Mp4Encoder.h"
#include "./../../Core/bitmap.h"
#include "./../../Tools/SystemTools.h"

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include "x264.h"

#include <filesystem>

ACMB_NAMESPACE_BEGIN

struct Mp4EncoderParams
{
    x264_param_t param;
    x264_picture_t pic;
    x264_picture_t pic_out;
    x264_t* h = nullptr;
    x264_nal_t* nal = nullptr;
    int i_nal = 0;
    int i_frame = 0;
};

Mp4Encoder::Mp4Encoder( H264Preset preset, H264Tune tune, H264Profile profile )
{
    _params = std::make_shared<Mp4EncoderParams>();

    if ( x264_param_default_preset( &_params->param, x264_preset_names[int( preset )], x264_tune_names[int( tune )] ) < 0 )
        throw std::runtime_error( "unable to create x264 encoder" );

    _params->param.i_bitdepth = 8;
    _params->param.i_csp = X264_CSP_RGB;
    _params->param.b_vfr_input = 0;
    _params->param.b_repeat_headers = 1;
    _params->param.b_annexb = 1;
    
    if ( x264_param_apply_profile( &_params->param, x264_profile_names[int( profile )] ) < 0 )
        throw std::runtime_error( "unable to create x264 encoder" );
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
    if ( _params->h )
    {
        while ( x264_encoder_delayed_frames( _params->h ) )
        {
            auto i_frame_size = x264_encoder_encode( _params->h, &_params->nal, &_params->i_nal, NULL, &_params->pic_out );
            if ( i_frame_size < 0 )
            {
                break;
            }
            else if ( i_frame_size )
            {
                if ( !fwrite( _params->nal->p_payload, i_frame_size, 1, _f ) )
                    break;
            }
        }

        x264_encoder_close( _params->h );
        _params->h = nullptr;

        x264_picture_clean( &_params->pic );
    }

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

    if ( !_params->h )
    {
        _params->param.i_width = pBitmap->GetWidth();
        _params->param.i_height = pBitmap->GetHeight();
        _params->h = x264_encoder_open( &_params->param );
        if ( !_params->h )
            throw std::runtime_error( "unable to create x264 encoder" );

        if ( x264_picture_alloc( &_params->pic, _params->param.i_csp, _params->param.i_width, _params->param.i_height ) < 0 )
            throw std::runtime_error( "unable to allocate x264 picture" );
    }

    if ( _params->param.i_width != pBitmap->GetWidth() || _params->param.i_height != pBitmap->GetHeight() )
        throw std::runtime_error( "bitmap size mismatch" );

    constexpr int channelCount = 3;

    tbb::parallel_for( tbb::blocked_range<int>( 0, _params->param.i_height ), [&] ( const tbb::blocked_range<int>& range )
    {
        for ( int i = range.begin(); i < range.end(); ++i )
        {
            auto pScanline = ( uint8_t* ) pBitmap->GetPlanarScanline( i );
            const int lineOffset = i * _params->pic.img.i_stride[0];
            for ( int j = 0; j < _params->param.i_width; ++j )
            {
                const int pixelOffset = lineOffset + j * channelCount;
                _params->pic.img.plane[0][pixelOffset] = pScanline[j * channelCount];
                _params->pic.img.plane[0][pixelOffset + 1] = pScanline[j * channelCount + 1];
                _params->pic.img.plane[0][pixelOffset + 2] = pScanline[j * channelCount + 2];
            }
        }
    } );

    _params->pic.i_pts = _params->i_frame++;
    auto i_frame_size = x264_encoder_encode( _params->h, &_params->nal, &_params->i_nal, &_params->pic, &_params->pic_out );
    if ( i_frame_size < 0 )
    {
        throw std::runtime_error( "unable to encode x264 frame" );
    }
    else if ( i_frame_size )
    {
        if ( !fwrite( _params->nal->p_payload, i_frame_size, 1, _f ) )
            throw std::runtime_error( "unable to encode x264 frame" );
    }
}

void Mp4Encoder::SetFrameRate( uint32_t rate )
{
    if ( _params )
        _params->param.i_fps_num = rate;
}

uint32_t Mp4Encoder::GetFrameRate() const
{
    return _params ? _params->param.i_fps_num : 0;
}

std::set<std::string> Mp4Encoder::GetExtensions()
{
    return { ".mp4", ".h264" };
}

ACMB_NAMESPACE_END