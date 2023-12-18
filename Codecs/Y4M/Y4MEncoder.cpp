#include "Y4MEncoder.h"
#include "./../../Core/bitmap.h"
#include "./../../Tools/SystemTools.h"

#include <filesystem>

ACMB_NAMESPACE_BEGIN

void Y4MEncoder::Attach( std::shared_ptr<std::ostream> pStream )
{
    if ( !pStream )
        throw std::invalid_argument( "pStream" );

    _pStream = pStream;
    _pStream->write( "YUV4MPEG2 ", 10 );
}

void Y4MEncoder::Detach()
{
    _pStream.reset();
    _yuv.clear();
}

void Y4MEncoder::WriteBitmap( std::shared_ptr<IBitmap> pBitmap )
{
    if ( !pBitmap )
        throw std::invalid_argument( "pBitmap is null" );

    if ( pBitmap->GetPixelFormat() != PixelFormat::RGB24 )
        throw std::invalid_argument( "unsupported pixel format" );

    if ( _width == 0 && _height == 0 )
    {
        _width = pBitmap->GetWidth();
        _height = pBitmap->GetHeight();
        
        *_pStream << "W" << _width << " ";
        *_pStream << "H" << _height << " ";
        *_pStream << "F" << _frameRate << ":1 ";
        *_pStream << "Ip ";
        *_pStream << "A1:1 ";
        *_pStream << "C420\x0a";      
        _yuv.resize( _width * _height / 2 * 3 );
    }

    if ( _width != pBitmap->GetWidth() || _height != pBitmap->GetHeight() )
        throw std::runtime_error( "bitmap size mismatch" );

    BitmapToYuv( pBitmap );

    _pStream->write( "FRAME\x0a", 6 );
    _pStream->write( (const char*)_yuv.data(), _yuv.size() );
}

std::set<std::string> Y4MEncoder::GetExtensions()
{
    return { ".y4m" };
}

ACMB_NAMESPACE_END