#include "TiffDecoder.h"
#include "../../Tools/SystemTools.h"

#include "tinytiffreader.hxx"
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <fstream>
#include <filesystem>

ACMB_NAMESPACE_BEGIN

template <PixelFormat pixelFormat>
void JoinChannels( std::shared_ptr<Bitmap<pixelFormat>> pBitmap, const uint8_t* data, int width, int height )
{
    constexpr auto channelCount = PixelFormatTraits<pixelFormat>::channelCount;
    using ChannelType = typename PixelFormatTraits<pixelFormat>::ChannelType;

    auto pInData = reinterpret_cast< const ChannelType* >( data );
    auto pOutData = pBitmap->GetScanline( 0 );

    oneapi::tbb::parallel_for( oneapi::tbb::blocked_range<int>( 0, height ), [&] ( const oneapi::tbb::blocked_range<int>& range )
    {
        for ( int i = range.begin(); i < range.end(); ++i )
        {
            for ( int j = 0; j < width; ++j )
            {
                for ( uint16_t ch = 0; ch < channelCount; ++ch )
                {
                    pOutData[( i * width + j ) * channelCount + ch] = pInData[ch * width * height + i * width + j];
                }
            }
        }
    } );
}

TiffDecoder::TiffDecoder( PixelFormat outputFormat )
: ImageDecoder( outputFormat )
{
}

void TiffDecoder::Attach( const std::string& fileName )
{
    _lastFileName = fileName;

    _pReader =  TinyTIFFReader_open( fileName.c_str() ) ;
    if ( !_pReader )
        throw std::runtime_error( "file is corrupted" );

    _width = TinyTIFFReader_getWidth(_pReader);
    _height = TinyTIFFReader_getHeight( _pReader );
    _decodedFormat = ConstructPixelFormat( TinyTIFFReader_getBitsPerSample( _pReader, 0 ), TinyTIFFReader_getSamplesPerPixel( _pReader ) );
    if ( _pixelFormat == PixelFormat::Bayer16 )
    {
        if ( _decodedFormat != PixelFormat::Gray16 )
            throw std::runtime_error( "unable to treat pixel format as bayer 16" );

        _decodedFormat = PixelFormat::Bayer16;
    }
    if ( _pixelFormat == PixelFormat::Unspecified )
        _pixelFormat = _decodedFormat;
}

void TiffDecoder::Attach( std::shared_ptr<std::istream> pStream )
{
    if ( !pStream )
        throw std::invalid_argument( "pStream" );
    
    _pStream = pStream;
    RandomStringGenerator rsg;
    const std::string fileName = std::filesystem::temp_directory_path().string() + "/tmp_" + rsg( 16 ) + ".tif";
    std::ofstream out( fileName, std::ios_base::out | std::ios_base::binary );
    if ( !out )
        throw std::runtime_error( "unable to open temporary file" );

    out << pStream->rdbuf();
    out.close();

    Attach( fileName );
}

void TiffDecoder::Detach()
{
    TinyTIFFReader_close( _pReader );
    if ( _pStream )
        std::filesystem::remove( _lastFileName );

    _pStream.reset();
}

IBitmapPtr TiffDecoder::ReadBitmap()
{
    if ( !_pReader )
        throw std::runtime_error( "TiffDecoder is detached" );
    IBitmapPtr pBitmap = IBitmap::Create( _width, _height, _decodedFormat );
    uint8_t* pData = nullptr;
    std::vector<uint8_t> data;
    if ( GetColorSpace( _decodedFormat ) == ColorSpace::Gray || GetColorSpace( _decodedFormat ) == ColorSpace::Bayer )
    {
        pData = (uint8_t*)pBitmap->GetPlanarScanline( 0 );
    }
    else
    {
        data.resize( _width * _height * BytesPerChannel( _decodedFormat ) * ChannelCount( _decodedFormat ) );
        pData = &data[0];
    }    
    
    uint32_t sampleSize = _width * _height * BytesPerChannel( _decodedFormat );
    for ( uint16_t i = 0; i < ChannelCount( _decodedFormat ); ++i )
    {
        TinyTIFFReader_getSampleData( _pReader, pData + i * sampleSize, i );
    }

    switch ( _decodedFormat )
    {
        case PixelFormat::RGB24:
            JoinChannels<PixelFormat::RGB24>( std::static_pointer_cast< Bitmap<PixelFormat::RGB24> >( pBitmap ), data.data(), _width, _height );
            break;
        case PixelFormat::RGB48:
            JoinChannels<PixelFormat::RGB48>( std::static_pointer_cast< Bitmap<PixelFormat::RGB48> >( pBitmap ), data.data(), _width, _height );
            break;
        default:
            break;
    }

    return ToOutputFormat(pBitmap);
}

std::unordered_set<std::string> TiffDecoder::GetExtensions()
{
    return { ".tiff", ".tif" };
}

ACMB_NAMESPACE_END