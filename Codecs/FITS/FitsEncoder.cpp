#include "FitsEncoder.h"
#include "../../Core/bitmap.h"
#include "../../Tools/SystemTools.h"

#include <CCfits/CCfits>
#include <filesystem>
#include <fstream>
#include <array>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

ACMB_NAMESPACE_BEGIN

void FitsEncoder::Attach( std::shared_ptr<std::ostream> pStream )
{
    if ( !pStream )
        throw std::invalid_argument( "pStream" );

    _pStream = pStream;
    RandomStringGenerator rsg;
    _fileName = std::filesystem::temp_directory_path().string() + "/tmp_" + rsg( 16 ) + ".tif";
    /*std::ofstream out(fileName, std::ios_base::out | std::ios_base::binary);
    if ( !out )
        throw std::runtime_error( "unable to open temporary file" );

    out << pStream->rdbuf();
    out.close();*/
}

void FitsEncoder::Attach( const std::string& fileName )
{
    _fileName = fileName;   
}

void FitsEncoder::Detach()
{
    _fileName.clear();
    _pStream.reset();
}

template<PixelFormat pixelFormat>
static void WriteBitmapToFits( std::shared_ptr<Bitmap<pixelFormat>> pBitmap, const std::string& fileName )
{
    constexpr bool isRGB = PixelFormatTraits<pixelFormat>::colorSpace == ColorSpace::RGB;
    constexpr long bitpix = PixelFormatTraits<pixelFormat>::bitsPerChannel == 8 ? BYTE_IMG : USHORT_IMG;    
    constexpr long naxis = isRGB ? 3 : 2;
    const long width = pBitmap->GetWidth();
    const long height = pBitmap->GetHeight();
    const long channelCount = PixelFormatTraits<pixelFormat>::channelCount;
    std::array<long, naxis> naxes = { width, height };
    if constexpr ( isRGB )
        naxes.back() = 3;

    std::unique_ptr<CCfits::FITS> pFits;

    try
    {
        pFits.reset( new CCfits::FITS( fileName, bitpix, naxis, naxes.data() ) );
    }
    catch ( CCfits::FitsException& e )
    {
        throw std::runtime_error( e.message() );
    }

    std::valarray<typename PixelFormatTraits<pixelFormat>::ChannelType> contents;
    const long nElements = width * height * channelCount;
    const long imageSize = width * height;
    contents.resize( nElements );

    tbb::parallel_for( tbb::blocked_range<long>( 0, height ), [&] ( const tbb::blocked_range<long>& range )
    {
        for ( long i = range.begin(); i < range.end(); ++i )
        {
            auto pScanline = (typename PixelFormatTraits<pixelFormat>::ChannelType*) (pBitmap->GetPlanarScanline( i ) );
            for ( long j = 0; j < width * channelCount; ++j )
            {
                const auto dv = std::div( j, channelCount );
                contents[imageSize * dv.rem + i * width + dv.quot] = pScanline[j];
            }
        }
    } );

    auto& header = pFits->pHDU();
    if ( isRGB )
        header.addKey( "CTYPE", "RGB     ", "");

    header.write( 1, nElements, contents );
}

void FitsEncoder::WriteBitmap( std::shared_ptr<IBitmap> pBitmap )
{
    if ( !pBitmap )
        throw std::invalid_argument( "pBitmap is null" );

    switch ( pBitmap->GetPixelFormat() )
    {
        case PixelFormat::Gray8:
            WriteBitmapToFits( std::static_pointer_cast< Bitmap<PixelFormat::Gray8> >(pBitmap), _fileName );
            break;
        case PixelFormat::Gray16:
            WriteBitmapToFits( std::static_pointer_cast< Bitmap<PixelFormat::Gray16> >(pBitmap), _fileName );
            break;
        case PixelFormat::Bayer16:
            WriteBitmapToFits( std::static_pointer_cast< Bitmap<PixelFormat::Bayer16> >(pBitmap), _fileName );
            break;
        case PixelFormat::RGB24:
            WriteBitmapToFits( std::static_pointer_cast< Bitmap<PixelFormat::RGB24> >(pBitmap), _fileName );
            break;
        case PixelFormat::RGB48:
            WriteBitmapToFits( std::static_pointer_cast< Bitmap<PixelFormat::RGB48> >(pBitmap), _fileName );
            break;
    }

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

std::set<std::string> FitsEncoder::GetExtensions()
{
    return { ".fit", ".fits" };
}

ACMB_NAMESPACE_END