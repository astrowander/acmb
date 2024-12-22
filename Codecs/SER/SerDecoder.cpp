#include "SerDecoder.h"
#include "../../Core/bitmap.h"

ACMB_NAMESPACE_BEGIN

SerDecoder::SerDecoder( PixelFormat outputFormat )
: ImageDecoder( outputFormat )
{
}

void SerDecoder::Attach( std::shared_ptr<std::istream> pStream )
{
    ImageDecoder::Attach( pStream );

    //check file size
    _pStream->seekg( 0, std::ios_base::end );
    uint64_t fileSize = _pStream->tellg();
    _pStream->seekg( 0, std::ios_base::beg );
    
    if ( fileSize < Ser::cHeaderSize )
        throw std::runtime_error( "invalid stream" );

    char signature[14];
    pStream->read( signature, 14 );    
    
    //skip unused bytes
    pStream->seekg(4, std::ios_base::cur );

    Ser::ColorID colorID;
    pStream->read( (char*)&colorID, 4 );

    int32_t littleEndian;
    pStream->read( (char*)&littleEndian, 4 );

    pStream->read( (char*)&_width, 4 );
    pStream->read( ( char* ) &_height, 4 );

    //_width = std::byteswap( header.imageWidth );
    //_height = header.imageHeight;
    int32_t pixelDepthPerPlane;
    pStream->read( (char*)&pixelDepthPerPlane, 4 );

    pStream->read( (char*)&_frameCount, 4 );

    if ( pixelDepthPerPlane != 8 && pixelDepthPerPlane != 16 )
        throw std::runtime_error( "unsupported pixel depth" );

    switch ( colorID )
    {
        case Ser::ColorID::MONO:
           pixelDepthPerPlane == 8 ? _decodedFormat = PixelFormat::Gray8 : _decodedFormat = PixelFormat::Gray16;
           break;
        case Ser::ColorID::BAYER_RGGB:
           if ( pixelDepthPerPlane != 16 )
               throw std::runtime_error( "unsupported pixel depth" );           
           _decodedFormat = PixelFormat::Bayer16;
           break;
        case Ser::ColorID::RGB:
            pixelDepthPerPlane == 8 ? _decodedFormat = PixelFormat::RGB24 : _decodedFormat = PixelFormat::RGB48;
           break;
        default:
            throw std::runtime_error( "unsupported colorID" );
            break;
    }

    if ( _pixelFormat == PixelFormat::Unspecified )
        _pixelFormat = _decodedFormat;

    _pCameraSettings = std::make_shared<CameraSettings>();

    pStream->read( _pCameraSettings->observer, 40 );
    pStream->read( _pCameraSettings->instrument, 40 );
    pStream->read( _pCameraSettings->telescope, 40 );

    pStream->read( (char*)&_pCameraSettings->timestamp, 8 );
    pStream->read( ( char* ) &_pCameraSettings->timestampUTC, 8 );

    _frameByteSize = _width * _height * BytesPerPixel( _decodedFormat );
}

std::shared_ptr<IBitmap> SerDecoder::ReadBitmap()
{
    if ( !_pStream )
        throw std::runtime_error( "Stream is not attached" );

    uint64_t oldPos = _pStream->tellg();
    auto pBitmap = IBitmap::Create( _width, _height, _decodedFormat );
    pBitmap->SetCameraSettings( _pCameraSettings );
    _pStream->read( pBitmap->GetPlanarScanline(0), _frameByteSize );

    if ( _pStream->tellg() != oldPos + _frameByteSize )
        throw std::runtime_error( "SERDecoder: attempt to read past end of stream" );

    return ToOutputFormat( pBitmap );
}

std::shared_ptr<IBitmap> SerDecoder::ReadBitmap( uint32_t i )
{
    if ( !_pStream )
        throw std::runtime_error( "Stream is not attached" );

    if ( i >= _frameCount )
        throw std::out_of_range( "Frame index out of range" );

    _pStream->seekg( Ser::cHeaderSize + i * _frameByteSize, std::ios_base::beg );
    return ReadBitmap();
}

std::unordered_set <std::string> SerDecoder::GetExtensions()
{
    return { ".ser" };
}

ACMB_NAMESPACE_END