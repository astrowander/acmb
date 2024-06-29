#include "SerEncoder.h"
#include "SerDecoder.h"
#include "./../../Core/bitmap.h"

ACMB_NAMESPACE_BEGIN

void SerEncoder::Attach( std::shared_ptr<std::ostream> pStream )
{
    if ( !pStream )
        throw std::invalid_argument( "pStream" );

    _pStream = pStream;
    _pStream->write( "LUCAM-RECORDER    ", 18 );
}

void SerEncoder::Detach()
{
    if ( _pStream )
    {
        _pStream->seekp( 38, std::ios::beg );
        _pStream->write( (char*)&_totalFrames, 4 );
        _pStream->seekp( 0, std::ios::end );
    }

    _pStream.reset();
}

void SerEncoder::WriteBitmap( std::shared_ptr<IBitmap> pBitmap )
{
    if ( !pBitmap )
        return;

    if ( _width == 0 && _height == 0 )
    {
        _width = pBitmap->GetWidth();
        _height = pBitmap->GetHeight();

        Ser::ColorID colorID{};

        switch ( pBitmap->GetPixelFormat() )
        {
        case PixelFormat::Gray8:
            _pixelFormat = PixelFormat::Gray8;
            colorID = Ser::ColorID::MONO;
            break;
        case PixelFormat::Gray16:
            _pixelFormat = PixelFormat::Gray16;
            colorID = Ser::ColorID::MONO;
            break;
        case PixelFormat::Bayer16:
            _pixelFormat = PixelFormat::Bayer16;
            colorID = Ser::ColorID::BAYER_RGGB;
            break;
        case PixelFormat::RGB24:
            _pixelFormat = PixelFormat::RGB24;
            colorID = Ser::ColorID::RGB;
            break;
        case PixelFormat::RGB48:
            _pixelFormat = PixelFormat::RGB48;
            colorID = Ser::ColorID::RGB;
            break;
        default:
            throw std::runtime_error( "unsupported pixel format" );
        }

        _pStream->write( (char*)&colorID, 4 );
        
        const int32_t littleEndian = 0;
        _pStream->write( (char*)&littleEndian, 4 );

        _pStream->write( (char*)&_width, 4 );
        _pStream->write( (char*)&_height, 4 );

        const int32_t pixelDepthPerPlane = BitsPerChannel( _pixelFormat );
        _pStream->write( (char*)&pixelDepthPerPlane, 4 );
        
        _pStream->write( (char*)&_totalFrames, 4 );

        auto pCameraSettings = pBitmap->GetCameraSettings();
        if ( pCameraSettings )
        {
            _pStream->write( pCameraSettings->telescope, 40 );
            _pStream->write( pCameraSettings->instrument, 40 );
            _pStream->write( pCameraSettings->observer, 40 );

            _pStream->write( ( char* ) &pCameraSettings->timestamp, 8 );
            _pStream->write( ( char* ) &pCameraSettings->timestampUTC, 8 );
        }
        else
        {
            const char zeroChars[136] = {};
            _pStream->write( zeroChars, 136 );
        }
    }

    if ( pBitmap->GetPixelFormat() != _pixelFormat )
        throw std::runtime_error( "pixel format mismatch" );

    if ( pBitmap->GetWidth() != _width || pBitmap->GetHeight() != _height )
        throw std::runtime_error( "image size mismatch" );

    const uint64_t byteSize = pBitmap->GetWidth() * pBitmap->GetHeight() * BytesPerPixel( _pixelFormat );
    _pStream->write( pBitmap->GetPlanarScanline(0), byteSize );

    ++_totalFrames;
}

std::set<std::string> SerEncoder::GetExtensions()
{
    return { ".ser" };
}

ACMB_NAMESPACE_END