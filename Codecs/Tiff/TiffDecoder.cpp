#include "TiffDecoder.h"


void TiffDecoder::Attach( const std::string& fileName )
{
    _pReader =  TinyTIFFReader_open( fileName.c_str() ) ;
    if ( !_pReader )
        throw std::runtime_error( "file is corrupted" );

    _width = TinyTIFFReader_getWidth(_pReader);
    _height = TinyTIFFReader_getHeight( _pReader );
    _pixelFormat = ConstructPixelFormat( TinyTIFFReader_getBitsPerSample( _pReader, 0 ), TinyTIFFReader_getSamplesPerPixel( _pReader ) );    
}

void TiffDecoder::Attach( std::shared_ptr<std::istream> )
{
    throw std::runtime_error( "not implemented" );
}

void TiffDecoder::Detach()
{
    TinyTIFFReader_close( _pReader );
}

IBitmapPtr TiffDecoder::ReadBitmap()
{
    if ( !_pReader )
        throw std::runtime_error( "TiffDecoder is detached" );
    IBitmapPtr pBitmap = IBitmap::Create( _width, _height, _pixelFormat );
    uint8_t* pData = nullptr;
    std::vector<uint8_t> data;
    if ( GetColorSpace( _pixelFormat ) == ColorSpace::Gray )
    {
        pData = (uint8_t*)pBitmap->GetPlanarScanline( 0 );
    }
    else
    {
        data.resize( _width * _height * BytesPerChannel( _pixelFormat ) * ChannelCount( _pixelFormat ) );
        pData = &data[0];
    }    
    
    uint32_t sampleSize = _width * _height * BytesPerChannel( _pixelFormat );    
    for ( uint16_t i = 0; i < ChannelCount( _pixelFormat ); ++i )
    {
        TinyTIFFReader_getSampleData( _pReader, pData + i * sampleSize, i );
    }

    switch ( _pixelFormat )
    {
        case PixelFormat::RGB24:
            JoinChannels<PixelFormat::RGB24>( std::static_pointer_cast< Bitmap<PixelFormat::RGB24> >( pBitmap ), data.data() );
            break;
        case PixelFormat::RGB48:
            JoinChannels<PixelFormat::RGB48>( std::static_pointer_cast< Bitmap<PixelFormat::RGB48> >( pBitmap ), data.data() );
            break;
        default:
            break;
    }

    return pBitmap;
}

std::shared_ptr<IBitmap> TiffDecoder::ReadStripe( uint32_t )
{
    throw std::runtime_error( "not implemented" );
}

uint32_t TiffDecoder::GetCurrentScanline() const
{
    throw std::runtime_error( "not implemented" );
}