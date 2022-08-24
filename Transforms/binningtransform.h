#pragma once
#include "basetransform.h"
#include "../Geometry/size.h"

class IBinningTransform : public BaseTransform
{
protected:
    Size _bin;
public:
    IBinningTransform(std::shared_ptr<IBitmap> pSrcBitmap, Size bin);

    static std::shared_ptr<IBinningTransform> Create(std::shared_ptr<IBitmap> pSrcBitmap, Size bin);
};

template<PixelFormat pixelFormat>
class BinningTransform final: public IBinningTransform
{
    using ChannelType = typename PixelFormatTraits<pixelFormat>::ChannelType;
    static constexpr auto channelCount = PixelFormatTraits<pixelFormat>::channelCount;

    std::vector<ChannelType> _buf;

public:
    BinningTransform(std::shared_ptr<IBitmap> pSrcBitmap, Size bin)
    : IBinningTransform(pSrcBitmap, bin)
    {
        _buf.resize(bin.width * bin.height * channelCount);
    }
    

    void Run() override
    {
        this->_pDstBitmap.reset(new Bitmap<pixelFormat>(_pSrcBitmap->GetWidth() / _bin.width, _pSrcBitmap->GetHeight() / _bin.height));
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
    void ProcessPixel(ChannelType* pSrcPixel, ChannelType* pDstPixel)
    {
        for (uint32_t ch = 0; ch < channelCount; ++ch)
        {
            double sum = 0.0;
            for (uint32_t i = 0; i < _bin.height; ++i)
            for (uint32_t j = 0; j < _bin.width; ++j)
            {
                sum += pSrcPixel[(this->_pSrcBitmap->GetWidth() * i + j) * channelCount + ch];
            }
            sum /= _bin.width * _bin.height;
            pDstPixel[ch] = static_cast<ChannelType>(std::clamp<double>(sum, 0, std::numeric_limits<ChannelType>::max()));
        }
    }

};

