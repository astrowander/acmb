#include "JpegDecoder.h"
#include "./TJpg_Decoder/tjpgd.h"

ACMB_NAMESPACE_BEGIN

JpegDecoder::JpegDecoder( PixelFormat outputPixelFormat )
: ImageDecoder( outputPixelFormat )
{}

void JpegDecoder::Attach( std::shared_ptr<std::istream> pStream )
{
    ImageDecoder::Attach( pStream );
    _decodedFormat = PixelFormat::RGB24;

    _pJdec = std::make_shared<JDEC>();
    _jdWorkspace.resize( TJPGD_WORKSPACE_SIZE );

    auto readBytes = [pStream = _pStream] ( JDEC*, uint8_t* buf, size_t len ) -> size_t
    {
        const size_t pos = pStream->tellg();

        buf ?
        pStream->read( ( char* ) buf, len ) :
        pStream->seekg( len, std::ios::cur );

        return size_t( pStream->tellg() ) - pos;
    };

    if ( jd_prepare( _pJdec.get(), readBytes, _jdWorkspace.data(), TJPGD_WORKSPACE_SIZE, 0 ) != JDR_OK )
        throw std::runtime_error( "Failed to initialize JDEC" );

    _width = _pJdec->width;
    _height = _pJdec->height;

    if ( _pixelFormat == PixelFormat::Unspecified )
        _pixelFormat = _decodedFormat;

    _startDataPos = pStream->tellg();
}

void JpegDecoder::Detach()
{
    ImageDecoder::Detach();
    _jdWorkspace.clear();
    _pJdec.reset();
    _pBitmap.reset();
}

std::shared_ptr<IBitmap> JpegDecoder::ReadBitmap()
{
    if ( !_pStream )
        throw std::runtime_error( "Stream is not attached" );

    _pStream->seekg( _startDataPos, std::ios_base::beg );
    _pBitmap = std::make_shared<Bitmap<PixelFormat::RGB24>>( _width, _height );

    auto readData = [pBitmap = _pBitmap] ( JDEC*, void* data, JRECT* rect ) -> int
    {
        const auto width = rect->right + 1 - rect->left;
        const auto height = rect->bottom + 1 - rect->top;
        const auto byteWidth = width * PixelFormatTraits<PixelFormat::RGB24>::channelCount;

        for ( int i = 0; i < height; ++i )
        {
            std::copy( ( uint8_t* ) data + i * byteWidth, ( uint8_t* ) data + (i + 1) * byteWidth, pBitmap->GetPlanarScanline( rect->top + i ) + PixelFormatTraits<PixelFormat::RGB24>::channelCount * rect->left );
        }

        return byteWidth * height;
    };

    if ( jd_decomp( _pJdec.get(), readData, 0 ) != JDR_OK )
        throw std::runtime_error( "Failed to decompress image" );

    return _pBitmap;
}

std::unordered_set<std::string> JpegDecoder::GetExtensions()
{
    return { ".jpg", ".jpeg", ".jfif" };
}

ACMB_NAMESPACE_END
