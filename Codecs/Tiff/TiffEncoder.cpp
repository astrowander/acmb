#include "TiffEncoder.h"
#include "tinytiffwriter.h"
#include "../../Core/bitmap.h"

ACMB_NAMESPACE_BEGIN

void TiffEncoder::Attach( std::shared_ptr<std::ostream> )
{
    throw std::runtime_error( "not implemented" );
}

void TiffEncoder::Attach( const std::string& fileName )
{
    _fileName = fileName;
}

void TiffEncoder::Detach()
{
    _fileName.clear();
}

void TiffEncoder::WriteBitmap( std::shared_ptr<IBitmap> pBitmap )
{
    if ( !pBitmap )
        throw std::invalid_argument( "pBitmap is null" );

    _pTiff = TinyTIFFWriter_open( _fileName.c_str(), BitsPerChannel( pBitmap->GetPixelFormat() ), TinyTIFFWriter_UInt, ChannelCount( pBitmap->GetPixelFormat() ), pBitmap->GetWidth(), pBitmap->GetHeight(), TinyTIFFWriter_AutodetectSampleInterpetation );
    if ( !_pTiff )
        throw std::runtime_error( "file writing error" );

    if ( !TinyTIFFWriter_writeImage( _pTiff, pBitmap->GetPlanarScanline( 0 ) ) )
        throw std::runtime_error( "file writing error" );

    TinyTIFFWriter_close( _pTiff );
}

ACMB_NAMESPACE_END