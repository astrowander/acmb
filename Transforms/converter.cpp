#include "converter.h"
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

template<PixelFormat srcPixelFormat, PixelFormat dstPixelFormat>
void ConvertPixel( std::conditional_t<BytesPerChannel( srcPixelFormat ) == 1, uint8_t*, uint16_t*>, std::conditional_t<BytesPerChannel( srcPixelFormat ) == 1, uint8_t*, uint16_t*> )
{
    throw std::runtime_error( "not implemented" );
}

template<>
void ConvertPixel<PixelFormat::RGB24, PixelFormat::Gray8>(uint8_t* pSrcPixel, uint8_t* pDstPixel)
{
    auto r = *pSrcPixel++;
    auto g = *pSrcPixel++;
    auto b = *pSrcPixel;

    *pDstPixel = static_cast<uint8_t>(0.299 * r + 0.587 * g + 0.114 * b + 0.5);
}

template<>
void ConvertPixel<PixelFormat::RGB48, PixelFormat::Gray16>(uint16_t* pSrcPixel, uint16_t* pDstPixel)
{
    auto r = *pSrcPixel++;
    auto g = *pSrcPixel++;
    auto b = *pSrcPixel;

    *pDstPixel = static_cast<uint16_t>(0.299 * r + 0.587 * g + 0.114 * b + 0.5);
}

BaseConverter::BaseConverter(IBitmapPtr pSrcBitmap)
: BaseTransform(pSrcBitmap)
{

}

std::shared_ptr<BaseConverter> BaseConverter::Create(IBitmapPtr pSrcBitmap, PixelFormat dstPixelFormat)
{
    if (!pSrcBitmap)
        throw std::invalid_argument("pSrcBitmap is null");

    if (pSrcBitmap->GetPixelFormat() == PixelFormat::RGB24 && dstPixelFormat == PixelFormat::Gray8)
    {
        return std::make_shared<Converter<PixelFormat::RGB24, PixelFormat::Gray8>>(pSrcBitmap);
    }
    else if (pSrcBitmap->GetPixelFormat() == PixelFormat::RGB48 && dstPixelFormat == PixelFormat::Gray16)
    {
        return std::make_shared<Converter<PixelFormat::RGB48, PixelFormat::Gray16>>(pSrcBitmap);
    }

    throw std::runtime_error("unsupported pixel format");
}

IBitmapPtr BaseConverter::Convert(IBitmapPtr pSrcBitmap, PixelFormat dstPixelFormat)
{
    return Create(pSrcBitmap, dstPixelFormat)->RunAndGetBitmap();
}

template<PixelFormat srcPixelFormat, PixelFormat dstPixelFormat>
Converter<srcPixelFormat, dstPixelFormat>::Converter( IBitmapPtr pSrcBitmap )
: BaseConverter( pSrcBitmap )
{

}

template<PixelFormat srcPixelFormat, PixelFormat dstPixelFormat>
void Converter<srcPixelFormat, dstPixelFormat>::Run()
{
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
