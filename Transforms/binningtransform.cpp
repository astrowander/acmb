#include "binningtransform.h"
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

ACMB_NAMESPACE_BEGIN

template<PixelFormat pixelFormat>
class BinningTransform_ final : public BinningTransform
{
    using ChannelType = typename PixelFormatTraits<pixelFormat>::ChannelType;
    static constexpr auto channelCount = PixelFormatTraits<pixelFormat>::channelCount;

    std::vector<ChannelType> _buf;

public:
    BinningTransform_( std::shared_ptr<IBitmap> pSrcBitmap, Size bin )
    : BinningTransform( pSrcBitmap, bin )
    {
        _buf.resize( bin.width * bin.height * channelCount );
    }

    virtual void Run() override
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


private:
    void ProcessPixel( ChannelType* pSrcPixel, ChannelType* pDstPixel )
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
            pDstPixel[ch] = static_cast< ChannelType >( std::clamp<double>( sum, 0, std::numeric_limits<ChannelType>::max() ) );
        }
    }
};


BinningTransform::BinningTransform(std::shared_ptr<IBitmap> pSrcBitmap, Size bin)
: BaseTransform(pSrcBitmap)
, _bin(bin)
{
    if (_bin.width == 0 || _bin.height == 0)
        throw std::invalid_argument("zero bin size");
}

std::shared_ptr<BinningTransform> BinningTransform::Create(std::shared_ptr<IBitmap> pSrcBitmap, Size bin)
{
    if (!pSrcBitmap)
        throw std::invalid_argument("pSrcBitmap is null");

    switch (pSrcBitmap->GetPixelFormat())
    {
    case PixelFormat::Gray8:
        return std::make_shared<BinningTransform_<PixelFormat::Gray8>>(pSrcBitmap, bin);
    case PixelFormat::Gray16:
        return std::make_shared<BinningTransform_<PixelFormat::Gray16>>(pSrcBitmap, bin);
    case PixelFormat::RGB24:
        return std::make_shared<BinningTransform_<PixelFormat::RGB24>>(pSrcBitmap, bin);
    case PixelFormat::RGB48:
        return std::make_shared<BinningTransform_<PixelFormat::RGB48>>(pSrcBitmap, bin);
    default:
        throw std::runtime_error("unsupported pixel format");
    }
}

std::shared_ptr<BinningTransform> BinningTransform::Create( PixelFormat pixelFormat, Size bin )
{
    switch ( pixelFormat )
    {
        case PixelFormat::Gray8:
            return std::make_shared<BinningTransform_<PixelFormat::Gray8>>( nullptr, bin );
        case PixelFormat::Gray16:
            return std::make_shared<BinningTransform_<PixelFormat::Gray16>>( nullptr, bin );
        case PixelFormat::RGB24:
            return std::make_shared<BinningTransform_<PixelFormat::RGB24>>( nullptr, bin );
        case PixelFormat::RGB48:
            return std::make_shared<BinningTransform_<PixelFormat::RGB48>>( nullptr, bin );
        default:
            throw std::runtime_error( "unsupported pixel format" );
    }
}

void BinningTransform::CalcParams( std::shared_ptr<ImageParams> pParams )
{
    _width = pParams->GetWidth() / _bin.width;
    _height = pParams->GetWidth() / _bin.height;
    _pixelFormat = pParams->GetPixelFormat();
}

ACMB_NAMESPACE_END