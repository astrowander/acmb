#include "binningtransform.h"
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

IBinningTransform::IBinningTransform(std::shared_ptr<IBitmap> pSrcBitmap, Size bin)
: BaseTransform(pSrcBitmap)
, _bin(bin)
{
    if (_bin.width == 0 || _bin.height == 0)
        throw std::invalid_argument("zero bin size");
}

std::shared_ptr<IBinningTransform> IBinningTransform::Create(std::shared_ptr<IBitmap> pSrcBitmap, Size bin)
{
    if (!pSrcBitmap)
        throw std::invalid_argument("pSrcBitmap is null");

    switch (pSrcBitmap->GetPixelFormat())
    {
    case PixelFormat::Gray8:
        return std::make_shared<BinningTransform<PixelFormat::Gray8>>(pSrcBitmap, bin);
    case PixelFormat::Gray16:
        return std::make_shared<BinningTransform<PixelFormat::Gray16>>(pSrcBitmap, bin);
    case PixelFormat::RGB24:
        return std::make_shared<BinningTransform<PixelFormat::RGB24>>(pSrcBitmap, bin);
    case PixelFormat::RGB48:
        return std::make_shared<BinningTransform<PixelFormat::RGB48>>(pSrcBitmap, bin);
    default:
        throw std::runtime_error("Pixel format must be known");
    }
}


template<PixelFormat pixelFormat>
BinningTransform<pixelFormat>::BinningTransform( std::shared_ptr<IBitmap> pSrcBitmap, Size bin )
: IBinningTransform( pSrcBitmap, bin )
{
    _buf.resize( bin.width * bin.height * channelCount );
}

template<PixelFormat pixelFormat>
void BinningTransform<pixelFormat>::Run()
{
    this->_pDstBitmap.reset( new Bitmap<pixelFormat>( _pSrcBitmap->GetWidth() / _bin.width, _pSrcBitmap->GetHeight() / _bin.height ) );
    auto pSrcBitmap = std::static_pointer_cast< Bitmap<pixelFormat> >( _pSrcBitmap );
    auto pDstBitmap = std::static_pointer_cast< Bitmap<pixelFormat> >( _pDstBitmap );

    oneapi::tbb::parallel_for( oneapi::tbb::blocked_range<int>( 0, _pSrcBitmap->GetHeight() / _bin.height ), [this, pSrcBitmap, pDstBitmap] ( const oneapi::tbb::blocked_range<int>& range )
    {
        for ( int i = range.begin(); i < range.end(); ++i )
        {
            auto pSrcPixel = pSrcBitmap->GetScanline( i * _bin.height );
            auto pDstPixel = pDstBitmap->GetScanline( i );

            for ( uint32_t j = 0; j < pDstBitmap->GetWidth(); ++j )
            {
                ProcessPixel( pSrcPixel, pDstPixel );
                pSrcPixel += channelCount * _bin.width;
                pDstPixel += channelCount;
            }
        }
    } );
}

template<PixelFormat pixelFormat>
void BinningTransform<pixelFormat>::ProcessPixel( ChannelType* pSrcPixel, ChannelType* pDstPixel )
{
    for ( uint32_t ch = 0; ch < channelCount; ++ch )
    {
        double sum = 0.0;
        for ( uint32_t i = 0; i < _bin.height; ++i )
            for ( uint32_t j = 0; j < _bin.width; ++j )
            {
                sum += pSrcPixel[( this->_pSrcBitmap->GetWidth() * i + j ) * channelCount + ch];
            }
        sum /= _bin.width * _bin.height;
        pDstPixel[ch] = static_cast<ChannelType>( std::clamp<double>( sum, 0, std::numeric_limits<ChannelType>::max() ) );
    }
}
