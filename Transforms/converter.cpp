#include "converter.h"
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#define CREATE_CONVERTER( fmt1, fmt2) \
if (pSrcBitmap->GetPixelFormat() == PixelFormat::fmt1 && dstPixelFormat == PixelFormat::fmt2)\
return std::make_shared<Converter_<PixelFormat::fmt1, PixelFormat::fmt2>>( pSrcBitmap );

#define CREATE_CONVERTER_FROM_FORMAT( fmt1, fmt2) \
if ( srcPixelFormat == PixelFormat::fmt1 && dstPixelFormat == PixelFormat::fmt2)\
return std::make_shared<Converter_<PixelFormat::fmt1, PixelFormat::fmt2>>( nullptr );

ACMB_NAMESPACE_BEGIN

template<PixelFormat srcPixelFormat, PixelFormat dstPixelFormat>
void ConvertPixel( std::conditional_t<BytesPerChannel( srcPixelFormat ) == 1, uint8_t*, uint16_t*>, std::conditional_t<BytesPerChannel( dstPixelFormat ) == 1, uint8_t*, uint16_t*> )
{
    throw std::runtime_error( "not implemented" );
}

template<>
void ConvertPixel<PixelFormat::RGB24, PixelFormat::Gray8>(uint8_t* pSrcChannel, uint8_t* pDstChannel)
{
    auto r = *pSrcChannel++;
    auto g = *pSrcChannel++;
    auto b = *pSrcChannel;

    *pDstChannel = static_cast<uint8_t>(0.299 * r + 0.587 * g + 0.114 * b + 0.5);
}

template<>
void ConvertPixel<PixelFormat::RGB24, PixelFormat::Gray16>( uint8_t* pSrcChannel, uint16_t* pDstChannel )
{
    auto r = *pSrcChannel++;
    auto g = *pSrcChannel++;
    auto b = *pSrcChannel;

    *pDstChannel = static_cast< uint16_t >( 0.299 * r + 0.587 * g + 0.114 * b + 0.5 ) << 8;
}

template<>
void ConvertPixel<PixelFormat::RGB24, PixelFormat::RGB48>( uint8_t* pSrcChannel, uint16_t* pDstChannel )
{
    *pDstChannel++ = *pSrcChannel++ << 8;
    *pDstChannel++ = *pSrcChannel++ << 8;
    *pDstChannel = *pSrcChannel << 8;
}

template<>
void ConvertPixel<PixelFormat::RGB48, PixelFormat::Gray16>(uint16_t* pSrcChannel, uint16_t* pDstChannel)
{
    auto r = *pSrcChannel++;
    auto g = *pSrcChannel++;
    auto b = *pSrcChannel;

    *pDstChannel = static_cast<uint16_t>(0.299 * r + 0.587 * g + 0.114 * b + 0.5);
}

template<>
void ConvertPixel<PixelFormat::RGB48, PixelFormat::Gray8>( uint16_t* pSrcChannel, uint8_t* pDstChannel )
{
    auto r = *pSrcChannel++;
    auto g = *pSrcChannel++;
    auto b = *pSrcChannel;

    *pDstChannel = static_cast< uint16_t >( 0.299 * r + 0.587 * g + 0.114 * b + 0.5 ) >> 8;
}

template<>
void ConvertPixel<PixelFormat::RGB48, PixelFormat::RGB24>( uint16_t* pSrcChannel, uint8_t* pDstChannel )
{
    *pDstChannel++ = *pSrcChannel++ >> 8;
    *pDstChannel++ = *pSrcChannel++ >> 8;
    *pDstChannel = *pSrcChannel >> 8;
}

template <>
void ConvertPixel<PixelFormat::Gray8, PixelFormat::Gray16>( uint8_t* pSrcChannel, uint16_t* pDstChannel )
{
    *pDstChannel = *pSrcChannel << 8;
}

template <>
void ConvertPixel<PixelFormat::Gray8, PixelFormat::RGB24>( uint8_t* pSrcChannel, uint8_t* pDstChannel )
{
    *pDstChannel++ = *pSrcChannel;
    *pDstChannel++ = *pSrcChannel;
    *pDstChannel = *pSrcChannel;
}

template <>
void ConvertPixel<PixelFormat::Gray8, PixelFormat::RGB48>( uint8_t* pSrcChannel, uint16_t* pDstChannel )
{
    const uint16_t dstChannel = *pSrcChannel << 8;
    *pDstChannel++ = dstChannel;
    *pDstChannel++ = dstChannel;
    *pDstChannel = dstChannel;
}

template <>
void ConvertPixel<PixelFormat::Gray16, PixelFormat::Gray8>( uint16_t* pSrcChannel, uint8_t* pDstChannel )
{
    *pDstChannel = *pSrcChannel >> 8;
}

template <>
void ConvertPixel<PixelFormat::Gray16, PixelFormat::RGB24>( uint16_t* pSrcChannel, uint8_t* pDstChannel )
{
    const uint8_t dstChannel = *pSrcChannel >> 8;
    *pDstChannel++ = dstChannel;
    *pDstChannel++ = dstChannel;
    *pDstChannel = dstChannel;
}

template <>
void ConvertPixel<PixelFormat::Gray16, PixelFormat::RGB48>( uint16_t* pSrcChannel, uint16_t* pDstChannel )
{
    *pDstChannel++ = *pSrcChannel;
    *pDstChannel++ = *pSrcChannel;
    *pDstChannel = *pSrcChannel;
}

template <PixelFormat srcPixelFormat, PixelFormat dstPixelFormat>
class Converter_ final : public Converter
{
public:
    Converter_( IBitmapPtr pSrcBitmap )
    : Converter( pSrcBitmap )
    {

    }

    virtual void Run() override
    {
        if ( srcPixelFormat == dstPixelFormat )
        {
            _pDstBitmap = _pSrcBitmap;
            return;
        }

        _pDstBitmap = std::make_shared<Bitmap<dstPixelFormat>>( _pSrcBitmap->GetWidth(), _pSrcBitmap->GetHeight() );
        auto pSrcBitmap = std::static_pointer_cast< Bitmap<srcPixelFormat> >( _pSrcBitmap );
        auto pDstBitmap = std::static_pointer_cast< Bitmap<dstPixelFormat> >( _pDstBitmap );
        oneapi::tbb::parallel_for( oneapi::tbb::blocked_range<int>( 0, _pSrcBitmap->GetHeight() ), [pSrcBitmap, pDstBitmap] ( const oneapi::tbb::blocked_range<int>& range )
        {
            for ( int i = range.begin(); i < range.end(); ++i )
            {
                auto pSrcScanline = pSrcBitmap->GetScanline( i );
                auto pDstScanline = pDstBitmap->GetScanline( i );

                for ( uint32_t j = 0; j < pDstBitmap->GetWidth(); ++j )
                {
                    ConvertPixel<srcPixelFormat, dstPixelFormat>( pSrcScanline, pDstScanline );
                    pSrcScanline += ChannelCount( srcPixelFormat );
                    pDstScanline += ChannelCount( dstPixelFormat );
                }
            }
        } );
    }

    virtual void CalcParams( std::shared_ptr<ImageParams> pParams ) override
    {
        _width = pParams->GetWidth();
        _height = pParams->GetHeight();
        _pixelFormat = dstPixelFormat;
    }
};

Converter::Converter(IBitmapPtr pSrcBitmap)
: BaseTransform(pSrcBitmap)
{

}

std::shared_ptr<Converter> Converter::Create(IBitmapPtr pSrcBitmap, PixelFormat dstPixelFormat)
{
    if (!pSrcBitmap)
        throw std::invalid_argument("pSrcBitmap is null");

    CREATE_CONVERTER( Gray8, Gray8 );
    CREATE_CONVERTER( Gray8, Gray16 );
    CREATE_CONVERTER( Gray8, RGB24 );
    CREATE_CONVERTER( Gray8, RGB48 );

    CREATE_CONVERTER( Gray16, Gray8 );
    CREATE_CONVERTER( Gray16, Gray16 );
    CREATE_CONVERTER( Gray16, RGB24 );
    CREATE_CONVERTER( Gray16, RGB48 );

    CREATE_CONVERTER( RGB24, Gray8 );
    CREATE_CONVERTER( RGB24, Gray16 );
    CREATE_CONVERTER( RGB24, RGB24 );
    CREATE_CONVERTER( RGB24, RGB48 );

    CREATE_CONVERTER( RGB48, Gray8 );
    CREATE_CONVERTER( RGB48, Gray16 );
    CREATE_CONVERTER( RGB48, RGB24 );
    CREATE_CONVERTER( RGB48, RGB48 );

    throw std::invalid_argument( "Unsupported pixel format" );
}

std::shared_ptr<Converter> Converter::Create( PixelFormat srcPixelFormat, PixelFormat dstPixelFormat )
{
    CREATE_CONVERTER_FROM_FORMAT( Gray8, Gray8 );
    CREATE_CONVERTER_FROM_FORMAT( Gray8, Gray16 );
    CREATE_CONVERTER_FROM_FORMAT( Gray8, RGB24 );
    CREATE_CONVERTER_FROM_FORMAT( Gray8, RGB48 );

    CREATE_CONVERTER_FROM_FORMAT( Gray16, Gray8 );
    CREATE_CONVERTER_FROM_FORMAT( Gray16, Gray16 );
    CREATE_CONVERTER_FROM_FORMAT( Gray16, RGB24 );
    CREATE_CONVERTER_FROM_FORMAT( Gray16, RGB48 );

    CREATE_CONVERTER_FROM_FORMAT( RGB24, Gray8 );
    CREATE_CONVERTER_FROM_FORMAT( RGB24, Gray16 );
    CREATE_CONVERTER_FROM_FORMAT( RGB24, RGB24 );
    CREATE_CONVERTER_FROM_FORMAT( RGB24, RGB48 );

    CREATE_CONVERTER_FROM_FORMAT( RGB48, Gray8 );
    CREATE_CONVERTER_FROM_FORMAT( RGB48, Gray16 );
    CREATE_CONVERTER_FROM_FORMAT( RGB48, RGB24 );
    CREATE_CONVERTER_FROM_FORMAT( RGB48, RGB48 );

    throw std::invalid_argument( "Unsupported pixel format" );
}

IBitmapPtr Converter::Convert(IBitmapPtr pSrcBitmap, PixelFormat dstPixelFormat)
{
    return Create(pSrcBitmap, dstPixelFormat)->RunAndGetBitmap();
}

ACMB_NAMESPACE_END
