#include "H265Encoder.h"
#include "./../../Core/bitmap.h"
#include "./../../Tools/BitmapTools.h"

#include "x265.h"

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/mem.h>
}

#include <filesystem>
#include <vector>
#include <cstring>
#include <cerrno>

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
    std::unique_ptr<AVFormatContext, decltype(&avformat_free_context)> pFormatContext{ nullptr, avformat_free_context };
    AVStream* pStream = nullptr;
    AVIOContext* pIoContext = nullptr;
    std::vector<uint8_t> extradata;
};

namespace
{
    constexpr int kDefaultAvioBufferSize = 32 * 1024;

    // Custom AVIO callback that forwards muxer writes straight into the existing std::ostream
    // so the rest of the code can keep using the ImageEncoder stream abstraction.
    int WritePacketToStream( void* opaque, const uint8_t* buf, int bufSize )
    {
        auto stream = static_cast<std::ostream*>( opaque );
        stream->write( reinterpret_cast<const char*>( buf ), bufSize );
        if ( !(*stream) )
            return AVERROR( EIO );

        return bufSize;
    }

    // Basic seek implementation for the std::ostream-backed AVIO context. The MP4 muxer only
    // needs forward progress and size queries, but we also support repositioning for safety.
    int64_t SeekStream( void* opaque, int64_t offset, int whence )
    {
        auto stream = static_cast<std::ostream*>( opaque );
        if ( whence == AVSEEK_SIZE )
        {
            const auto current = stream->tellp();
            stream->seekp( 0, std::ios::end );
            const auto end = stream->tellp();
            stream->seekp( current );
            return static_cast<int64_t>( end );
        }

        std::ios_base::seekdir dir = std::ios::beg;
        switch ( whence )
        {
        case SEEK_CUR: dir = std::ios::cur; break;
        case SEEK_END: dir = std::ios::end; break;
        case SEEK_SET: dir = std::ios::beg; break;
        default: return AVERROR( EINVAL );
        }

        stream->seekp( offset, dir );
        if ( !(*stream) )
            return AVERROR( EIO );

        return static_cast<int64_t>( stream->tellp() );
    }
}

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

void H265Encoder::Attach( const std::string& fileName )
{
    const auto extension = std::filesystem::path( fileName ).extension().string();
    _useMp4 = ( extension == ".mp4" );
    ImageEncoder::Attach( fileName );
}

void H265Encoder::Detach()
{
    if ( _params->pEncoder )
    {
        // Flush delayed frames from the encoder. Each encode call without input will emit
        // any buffered frames so the muxer receives a complete sequence before finalization.
        while ( x265_encoder_encode( _params->pEncoder.get(), &_params->pNal, &_params->iNal, nullptr, _params->pPicOut.get() ) > 0 )
        {
            std::vector<uint8_t> packetData;
            size_t totalSize = 0;
            for ( uint32_t i = 0; i < _params->iNal; ++i )
                totalSize += _params->pNal[i].sizeBytes;

            packetData.resize( totalSize );
            size_t offset = 0;
            for ( uint32_t i = 0; i < _params->iNal; ++i )
            {
                memcpy( packetData.data() + offset, _params->pNal[i].payload, _params->pNal[i].sizeBytes );
                offset += _params->pNal[i].sizeBytes;
            }

            if ( _useMp4 && _params->pFormatContext && _params->pStream )
            {
                AVPacket* packet = av_packet_alloc();
                if ( !packet )
                    throw std::runtime_error( "unable to allocate packet" );

                // av_interleaved_write_frame takes ownership and will free this buffer.
                packet->data = static_cast<uint8_t*>( av_memdup( packetData.data(), packetData.size() ) );
                packet->size = static_cast<int>( packetData.size() );
                packet->stream_index = _params->pStream->index;
                packet->pts = packet->dts = _params->pPicOut->pts;
                packet->duration = 1;

                AVRational encoderTimeBase{ 1, static_cast<int>( _frameRate ) };
                av_packet_rescale_ts( packet, encoderTimeBase, _params->pStream->time_base );
                av_interleaved_write_frame( _params->pFormatContext.get(), packet );
                av_packet_free( &packet );
            }
            else if ( _pStream )
            {
                _pStream->write( reinterpret_cast<const char*>( packetData.data() ), packetData.size() );
            }
        }
    }

    if ( _useMp4 && _params->pFormatContext )
    {
        // Ensure the MP4 headers and trailer are fully written so the resulting file is playable.
        av_write_trailer( _params->pFormatContext.get() );
        if ( _params->pIoContext )
        {
            av_freep( &_params->pIoContext->buffer );
            avio_context_free( &_params->pIoContext );
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

        if ( _useMp4 )
        {
            uint32_t headerCount = 0;
            x265_nal* headerNal = nullptr;
            // The HEVC decoder configuration (VPS/SPS/PPS) must be captured up front to populate
            // the MP4 extradata before the first frame is written.
            int headerSize = x265_encoder_headers( _params->pEncoder.get(), &headerNal, &headerCount );
            if ( headerSize < 0 )
                throw std::runtime_error( "unable to get x265 headers" );

            size_t extradataSize = 0;
            for ( uint32_t i = 0; i < headerCount; ++i )
                extradataSize += headerNal[i].sizeBytes;

            _params->extradata.resize( extradataSize );
            size_t offset = 0;
            for ( uint32_t i = 0; i < headerCount; ++i )
            {
                memcpy( _params->extradata.data() + offset, headerNal[i].payload, headerNal[i].sizeBytes );
                offset += headerNal[i].sizeBytes;
            }

            AVFormatContext* pFormat = nullptr;
            if ( avformat_alloc_output_context2( &pFormat, nullptr, "mp4", nullptr ) < 0 || !pFormat )
                throw std::runtime_error( "unable to create mp4 muxer" );

            _params->pFormatContext.reset( pFormat );
            _params->pStream = avformat_new_stream( _params->pFormatContext.get(), nullptr );
            if ( !_params->pStream )
                throw std::runtime_error( "unable to create video stream" );

            _params->pStream->codecpar->codec_type = AVMEDIA_TYPE_VIDEO;
            _params->pStream->codecpar->codec_id = AV_CODEC_ID_HEVC;
            _params->pStream->codecpar->width = _width;
            _params->pStream->codecpar->height = _height;
            _params->pStream->codecpar->format = AV_PIX_FMT_YUV420P;
            _params->pStream->codecpar->extradata_size = static_cast<int>( _params->extradata.size() );
            _params->pStream->codecpar->extradata = static_cast<uint8_t*>( av_memdup( _params->extradata.data(), _params->extradata.size() ) );
            _params->pStream->time_base = { 1, static_cast<int>( _frameRate ) };

            auto buffer = static_cast<unsigned char*>( av_malloc( kDefaultAvioBufferSize ) );
            if ( !buffer )
                throw std::runtime_error( "unable to allocate avio buffer" );

            // Wire libavformat up to our std::ostream so callers do not need to manage a file
            // descriptor or custom URL protocol.
            _params->pIoContext = avio_alloc_context( buffer, kDefaultAvioBufferSize, 1, _pStream.get(), nullptr, &WritePacketToStream, &SeekStream );
            if ( !_params->pIoContext )
            {
                av_freep( &buffer );
                throw std::runtime_error( "unable to create avio context" );
            }

            _params->pFormatContext->pb = _params->pIoContext;
            _params->pFormatContext->flags |= AVFMT_FLAG_CUSTOM_IO;

            if ( avformat_write_header( _params->pFormatContext.get(), nullptr ) < 0 )
                throw std::runtime_error( "unable to write mp4 header" );
        }
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
        std::vector<uint8_t> packetData;
        size_t totalSize = 0;
        for ( uint32_t i = 0; i < _params->iNal; ++i )
            totalSize += _params->pNal[i].sizeBytes;

        packetData.resize( totalSize );
        size_t offset = 0;
        for ( uint32_t i = 0; i < _params->iNal; ++i )
        {
            memcpy( packetData.data() + offset, _params->pNal[i].payload, _params->pNal[i].sizeBytes );
            offset += _params->pNal[i].sizeBytes;
        }

        if ( _useMp4 && _params->pFormatContext && _params->pStream )
        {
            AVPacket* packet = av_packet_alloc();
            if ( !packet )
                throw std::runtime_error( "unable to allocate packet" );

            // av_interleaved_write_frame takes ownership and will free this buffer.
            packet->data = static_cast<uint8_t*>( av_memdup( packetData.data(), packetData.size() ) );
            packet->size = static_cast<int>( packetData.size() );
            packet->stream_index = _params->pStream->index;
            packet->pts = packet->dts = _params->pPicOut->pts;
            packet->duration = 1;

            AVRational encoderTimeBase{ 1, static_cast<int>( _frameRate ) };
            av_packet_rescale_ts( packet, encoderTimeBase, _params->pStream->time_base );
            av_interleaved_write_frame( _params->pFormatContext.get(), packet );
            av_packet_free( &packet );
        }
        else
        {
            _pStream->write( reinterpret_cast<const char*>( packetData.data() ), packetData.size() );
        }
    }
}

std::set<std::string> H265Encoder::GetExtensions()
{
    return { ".mp4", ".h265" };
}


ACMB_NAMESPACE_END