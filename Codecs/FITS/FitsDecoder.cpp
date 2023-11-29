#include "FitsDecoder.h"
#include "../../Tools/SystemTools.h"

#include <CCfits/CCfits>
#include <fstream>
#include <filesystem>
#include <valarray>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

ACMB_NAMESPACE_BEGIN

FitsDecoder::FitsDecoder( PixelFormat outputFormat )
: ImageDecoder( outputFormat )
{
}

void FitsDecoder::Attach( const std::string& fileName )
{
    _lastFileName = fileName;
    _pFits = std::make_unique<CCfits::FITS>( fileName, CCfits::RWmode::Read, true );

    auto& header = _pFits->pHDU();
    header.readAllKeys();

    const auto bitpix = header.bitpix();
    const auto axisCount = header.axes();

    if ( axisCount != 2 && axisCount != 3 )
        throw std::runtime_error( "unsupported FITS format" );

    _width = header.axis( 0 );
    _height = header.axis( 1 );

    switch ( bitpix )
    {
        case CCfits::Ibyte:
            _decodedFormat = (axisCount == 2) ? PixelFormat::Gray8 : PixelFormat::RGB24;
            break;
        case CCfits::Ishort:
        case CCfits::Ifloat:
            _decodedFormat = (axisCount == 2) ? PixelFormat::Gray16 : PixelFormat::RGB48;
            break;
        default:
            throw std::runtime_error( "unsupported FITS format" );
    }

    if ( _pixelFormat == PixelFormat::Bayer16 )
    {
        if ( _decodedFormat != PixelFormat::Gray16 )
            throw std::runtime_error( "unable to treat pixel format as bayer 16" );

        _decodedFormat = PixelFormat::Bayer16;
    }
    if ( _pixelFormat == PixelFormat::Unspecified )
        _pixelFormat = _decodedFormat;
}

void FitsDecoder::Attach( std::shared_ptr<std::istream> pStream )
{
    if ( !pStream )
        throw std::invalid_argument( "pStream is null" );

    _pStream = pStream;
    RandomStringGenerator rsg;
    const std::string fileName = std::filesystem::temp_directory_path().string() + "/tmp_" + rsg( 16 ) + ".fit";
    std::ofstream out( fileName, std::ios_base::out | std::ios_base::binary );
    if ( !out )
        throw std::runtime_error( "unable to open temporary file" );

    out << pStream->rdbuf();
    out.close();

    Attach( fileName );
}

void FitsDecoder::Detach()
{
    if ( _pStream )
        std::filesystem::remove( _lastFileName );

    _pFits.reset();
    _pStream.reset();
}

IBitmapPtr FitsDecoder::ReadBitmap()
{
    if ( !_pFits )
        throw std::runtime_error( "FitsDecoder is detached" );

    IBitmapPtr pBitmap = IBitmap::Create( _width, _height, _decodedFormat );

    auto& header = _pFits->pHDU();
    if ( BytesPerChannel( _decodedFormat ) == 1 )
    {
        std::valarray<uint8_t> contents;
        header.read( contents );
        std::copy( std::begin( contents ), std::end( contents ), pBitmap->GetPlanarScanline( 0 ) );
    }
    else if ( header.bitpix() == CCfits::Ishort )
    {
        std::valarray<uint16_t> contents;
        header.read( contents );
        std::copy( std::begin( contents ), std::end( contents ), pBitmap->GetPlanarScanline( 0 ) );
    }
    else if ( header.bitpix() == CCfits::Ifloat )
    {
        std::valarray<float> contents;
        header.read( contents );
        const int channelCount = ChannelCount( _pixelFormat );
        const int imageSize = _width * _height;

        tbb::parallel_for( tbb::blocked_range<int>( 0, _height ), [&] ( const tbb::blocked_range<int>& range )
        {
            for ( int i = range.begin(); i < range.end(); ++i )
            {
                uint16_t* pScanline = ( uint16_t* ) (pBitmap->GetPlanarScanline( i ));
                for ( int j = 0; j < _width * channelCount; ++j )
                {
                    const auto dv = std::div( j, channelCount );
                    pScanline[j] = uint16_t( std::clamp( contents[imageSize * dv.rem + i * _width + dv.quot] + 0.5f, 0.f, 65535.f ) );
                }
            }
        } );
    }

    return ToOutputFormat( pBitmap );
}

std::unordered_set<std::string> FitsDecoder::GetExtensions()
{
    return { ".fits", ".fit" };
}

ACMB_NAMESPACE_END
