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
    
    if ( fileSize < cHeaderSize )
        throw std::runtime_error( "invalid stream" );

    char signature[14];
    pStream->read( signature, 14 );    
    
    //skip unused bytes
    pStream->seekg(4, std::ios_base::cur );

    ColorID colorID;
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

    switch ( colorID )
    {
        case ColorID::MONO:
           pixelDepthPerPlane == 8 ? _decodedFormat = PixelFormat::Gray8 : _decodedFormat = PixelFormat::Gray16;
           break;
        case ColorID::BAYER_RGGB:
           if ( pixelDepthPerPlane != 16 )
               throw std::runtime_error( "unsupported pixel depth" );           
           _decodedFormat = PixelFormat::Bayer16;
           break;
        case ColorID::RGB:
           _decodedFormat = PixelFormat::RGB24;
           break;
        default:
            throw std::runtime_error( "unsupported colorID" );
            break;
    }

    if ( _pixelFormat == PixelFormat::Unspecified )
        _pixelFormat = _decodedFormat;

    pStream->seekg( cHeaderSize, std::ios_base::beg );
}

std::shared_ptr<IBitmap> SerDecoder::ReadBitmap()
{
    if ( !_pStream )
        throw std::runtime_error( "Stream is not attached" );

    uint64_t oldPos = _pStream->tellg();
    auto pBitmap = IBitmap::Create( _width, _height, _decodedFormat );
    const uint64_t byteSize = _width * _height * BytesPerPixel( _decodedFormat );
    _pStream->read( pBitmap->GetPlanarScanline(0), byteSize );

    if ( _pStream->tellg() != oldPos + byteSize )
        throw std::runtime_error( "invalid stream" );

    return pBitmap;
}

std::unordered_set <std::string> SerDecoder::GetExtensions()
{
    return { ".ser" };
}

ACMB_NAMESPACE_END