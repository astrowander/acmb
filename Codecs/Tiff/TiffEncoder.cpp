#include "TiffEncoder.h"
#include "tinytiffwriter.h"
#include "../../Core/bitmap.h"
#include "../../Tools/SystemTools.h"

#include <filesystem>
#include <fstream>

ACMB_NAMESPACE_BEGIN


void TiffEncoder::Attach( std::shared_ptr<std::ostream> pStream )
{
    if ( !pStream )
        throw std::invalid_argument( "pStream" );

    _pStream = pStream;
}

void TiffEncoder::Attach( const std::string& fileName )
{
    _fileName = fileName;
}

void TiffEncoder::Detach()
{
    _fileName.clear();
    _pStream.reset();
}

void TiffEncoder::WriteBitmap( std::shared_ptr<IBitmap> pBitmap )
{
    if ( !pBitmap )
        throw std::invalid_argument( "pBitmap is null" );

    if ( _pStream )
    {
        RandomStringGenerator rsg;
        _fileName = std::filesystem::temp_directory_path().string() + "/tmp_" + rsg( 16 ) + ".tif";
    }

    _pTiff = TinyTIFFWriter_open( _fileName.c_str(), BitsPerChannel( pBitmap->GetPixelFormat() ), TinyTIFFWriter_UInt, ChannelCount( pBitmap->GetPixelFormat() ), pBitmap->GetWidth(), pBitmap->GetHeight(), TinyTIFFWriter_AutodetectSampleInterpetation );
    if ( !_pTiff )
        throw std::runtime_error( "file writing error" );

    if ( !TinyTIFFWriter_writeImage( _pTiff, pBitmap->GetPlanarScanline( 0 ) ) )
        throw std::runtime_error( "file writing error" );

    TinyTIFFWriter_close( _pTiff );

    if ( _pStream )
    {
        std::ifstream in( _fileName, std::ios_base::in | std::ios_base::binary );
        if ( !in )
            throw std::runtime_error( "unable to open temporary file" );

        *_pStream << in.rdbuf();
        in.close();
        std::filesystem::remove( _fileName );
    }
}

std::unordered_set<std::string> TiffEncoder::GetExtensions()
{
    return { ".tif", ".tiff" };
}

ACMB_NAMESPACE_END