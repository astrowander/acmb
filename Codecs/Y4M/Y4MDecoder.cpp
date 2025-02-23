#include "Y4MDecoder.h"
#include "../../Core/bitmap.h"
#include "../../Tools/BitmapTools.h"

#include <charconv>

ACMB_NAMESPACE_BEGIN

/*oid BitmapFromYUV(std::shared_ptr<Bitmap<PixelFormat::RGB24>> pBitmap, const std::vector<uint8_t>& yuv)
{
    if ( !pBitmap )
        throw std::invalid_argument( "pBitmap is null" );

    const auto width = pBitmap->GetWidth();
    const auto height = pBitmap->GetHeight();

    const uint8_t* yPlane = yuv.data();
    const uint8_t* uPlane = yPlane + width * height;
    const uint8_t* vPlane = uPlane + width * height / 4;

    constexpr int channelCount = 3;

    for ( uint32_t line = 0; line < height; line++ )
    {
        auto pPixel = ( uint8_t* ) pBitmap->GetPlanarScanline( line );
        const size_t lPos = line * width;

        const size_t uPos = line * width / 4;
        const size_t vPos = line * width / 4;

        for ( uint32_t x = 0; x < width; x++ )
        {
            const uint8_t y = yPlane[lPos + x];
            const uint8_t u = uPlane[uPos + x / 2];
            const uint8_t v = vPlane[vPos + x / 2];

            const float c = y - 16;
            const float d = u - 128;
            const float e = v - 128;

            pPixel[0] = uint8_t( std::clamp<float>( 1.164f * c + 1.596f * e, 0.0f, 255.0f ) + 0.5f );
            pPixel[1] = uint8_t( std::clamp<float>( 1.164f * c - 0.391f * d - 0.813f * e, 0.0f, 255.0f ) + 0.5f );
            pPixel[2] = uint8_t( std::clamp<float>( 1.164f * c + 2.018f * d, 0.0f, 255.0f ) + 0.5f );
            pPixel += channelCount;
        }
    }
}*/

Y4MDecoder::Y4MDecoder( PixelFormat outputFormat )
: ImageDecoder( outputFormat ) 
{
}

void Y4MDecoder::Attach( std::shared_ptr<std::istream> pStream )
{
    ImageDecoder::Attach( pStream );

    //check file size
    _pStream->seekg( 0, std::ios_base::end );
    uint64_t fileSize = _pStream->tellg();
    _pStream->seekg( 0, std::ios_base::beg );

    if ( fileSize < Y4M::cSignatureSize )
        throw std::runtime_error( "Y4MDecoder: invalid stream" );

    char signature[10] = {};
    pStream->read( signature, 9 );
    if ( std::string( signature, 9 ) != "YUV4MPEG2" )
        throw std::runtime_error( "Y4MDecoder: invalid signature" );

    char nextChar = pStream->get();
    if ( nextChar != ' ' )
        throw std::runtime_error( "Y4MDecoder: invalid stream" );

    while ( nextChar != 0x0A )
    {
        while ( nextChar == ' ' )
            nextChar = pStream->get();

        const char key = nextChar;
        nextChar = pStream->get();

        std::string data;
        while ( nextChar != ' ' && nextChar != 0x0A )
        {
            data.push_back( nextChar );
            nextChar = pStream->get();
        }

        uint32_t value = 0;
        std::from_chars( data.c_str(), data.c_str() + data.size(), value );

        switch ( key )
        {
            case 'W':
                _width = value;
                break;
            case 'H':
                _height = value;
                break;
            case 'F':
                _frameRate = value;
                break;
            case 'C':
                if ( value != 420 )
                    throw std::runtime_error( "Y4MDecoder: unsupported color format" );
                break;
        }
    }

    if ( _width == 0 || _height == 0 || _frameRate == 0 )
        throw std::runtime_error( "Y4MDecoder:  stream" );

    _headerSize = pStream->tellg();
    _frameBufferSize = _width * _height * 3 / 2;
    _frameByteSize = _frameBufferSize + 6;

    const size_t rest = fileSize - _headerSize;
    _frameCount = uint32_t( rest / _frameByteSize );
    if ( _frameCount == 0 || rest % _frameByteSize != 0 )
        throw std::runtime_error( "Y4MDecoder: invalid stream" );
}

std::shared_ptr<IBitmap> Y4MDecoder::ReadBitmap()
{
    if ( !_pStream )
        throw std::runtime_error( "Y4MDecoder: decoder is detached" );

    auto pBitmap = std::make_shared<Bitmap<PixelFormat::YUV24>>( _width, _height );
    uint64_t oldPos = _pStream->tellg();
    char signature[7] = {};
    _pStream->read( signature, 6 );
    if ( std::string( signature, 6 ) != "FRAME\x0a" )
        throw std::runtime_error( "Y4MDecoder: invalid stream" );

    std::vector<uint8_t> buffer( _frameBufferSize );
    _pStream->read( (char*)buffer.data(), _frameBufferSize );

    if ( _pStream->tellg() != oldPos + _frameByteSize )
        throw std::runtime_error( "Y4MDecoder: attempt to read past end of stream" );

    return YUVBitmapFromPlanarData( buffer, _width, _height );
}

std::shared_ptr<IBitmap> Y4MDecoder::ReadBitmap( uint32_t i )
{
    if ( !_pStream )
        throw std::runtime_error( "Stream is not attached" );

    if ( i >= _frameCount )
        throw std::out_of_range( "Frame index out of range" );

    _pStream->seekg( _headerSize + i * _frameByteSize, std::ios_base::beg );
    return ReadBitmap();
}

std::unordered_set <std::string> Y4MDecoder::GetExtensions()
{
    return { ".y4m", ".yuv" };
}

ACMB_NAMESPACE_END