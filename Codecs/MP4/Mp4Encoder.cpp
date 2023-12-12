#include "Mp4Encoder.h"
#include "./../../Core/bitmap.h"
#include "./../../Tools/SystemTools.h"

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <filesystem>

ACMB_NAMESPACE_BEGIN

Mp4Encoder::Mp4Encoder( H264Preset preset, H264Tune tune, H264Profile profile )
{
    if ( x264_param_default_preset( &_param, x264_preset_names[int( preset )], x264_tune_names[int( tune )] ) < 0 )
        throw std::runtime_error( "unable to create x264 encoder" );

    _param.i_bitdepth = 8;
    _param.i_csp = X264_CSP_RGB;
    _param.b_vfr_input = 0;
    _param.b_repeat_headers = 1;
    _param.b_annexb = 1;

    if ( x264_param_apply_profile( &_param, x264_profile_names[int( profile )] ) < 0 )
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
    _i_frame = 0;
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
    if ( _h )
    {
        while ( x264_encoder_delayed_frames( _h ) )
        {
            auto i_frame_size = x264_encoder_encode( _h, &_nal, &i_nal, NULL, &_pic_out );
            if ( i_frame_size < 0 )
            {
                break;
            }
            else if ( i_frame_size )
            {
                if ( !fwrite( _nal->p_payload, i_frame_size, 1, _f ) )
                    break;
            }
        }

        x264_encoder_close( _h );
        _h = nullptr;

        x264_picture_clean( &_pic );
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

    if ( !_h )
    {
        _param.i_width = pBitmap->GetWidth();
        _param.i_height = pBitmap->GetHeight();
        _h = x264_encoder_open( &_param );
        if ( !_h )
            throw std::runtime_error( "unable to create x264 encoder" );

        if ( x264_picture_alloc( &_pic, _param.i_csp, _param.i_width, _param.i_height ) < 0 )
            throw std::runtime_error( "unable to allocate x264 picture" );
    }

    constexpr int channelCount = 3;

    tbb::parallel_for( tbb::blocked_range<int>( 0, _param.i_height ), [&] ( const tbb::blocked_range<int>& range )
    {
        for ( int i = range.begin(); i < range.end(); ++i )
        {
            auto pScanline = ( uint8_t* ) pBitmap->GetPlanarScanline( i );
            for ( int j = 0; j < _param.i_width; ++j )
            {
                _pic.img.plane[0][i * _param.i_width + j] = pScanline[j * channelCount];
                _pic.img.plane[1][i * _param.i_width + j] = pScanline[j * channelCount + 1];
                _pic.img.plane[2][i * _param.i_width + j] = pScanline[j * channelCount + 2];
            }
        }
    } );

    _pic.i_pts = _i_frame++;
    auto i_frame_size = x264_encoder_encode( _h, &_nal, &i_nal, &_pic, &_pic_out );
    if ( i_frame_size < 0 )
    {
        throw std::runtime_error( "unable to encode x264 frame" );
    }
    else if ( i_frame_size )
    {
        if ( !fwrite( _nal->p_payload, i_frame_size, 1, _f ) )
            throw std::runtime_error( "unable to encode x264 frame" );
    }
}

std::set<std::string> Mp4Encoder::GetExtensions()
{
    return { ".mp4", ".h264" };
}

ACMB_NAMESPACE_END