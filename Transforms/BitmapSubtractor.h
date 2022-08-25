#pragma once
#include "basetransform.h"

class BaseBitmapSubtractor : public BaseTransform
{
protected:
    IBitmapPtr _pBitmapToSubtract;

public:
    BaseBitmapSubtractor( IBitmapPtr pSrcBitmap, IBitmapPtr pBitmapToSubtract );
    static std::shared_ptr<BaseBitmapSubtractor> Create( IBitmapPtr pSrcBitmap, IBitmapPtr pBitmapToSubtract );
    static IBitmapPtr Subtract( IBitmapPtr pSrcBitmap, IBitmapPtr pBitmapToSubtract );
};

template <PixelFormat pixelFormat>
class BitmapSubtractor final: public BaseBitmapSubtractor
{
    using ChannelType = typename PixelFormatTraits<pixelFormat>::ChannelType;

public:

    BitmapSubtractor( IBitmapPtr pSrcBitmap, IBitmapPtr pBitmapToSubtract )
    : BaseBitmapSubtractor(pSrcBitmap, pBitmapToSubtract)    
    { }

    void Run() override
    {
        auto pSrcBitmap = std::static_pointer_cast< Bitmap<pixelFormat> >( _pSrcBitmap );
        auto pBitmapToSubtract = std::static_pointer_cast< Bitmap<pixelFormat> >( _pBitmapToSubtract );

        oneapi::tbb::parallel_for( oneapi::tbb::blocked_range<int>( 0, _pSrcBitmap->GetHeight() ), [pSrcBitmap, pBitmapToSubtract] (const oneapi::tbb::blocked_range<int>& range)
        {           
            for ( int i = range.begin(); i < range.end(); ++i )
            {
                auto pSrcScanline = pSrcBitmap->GetScanline( i );
                auto pScanlineToSubtract = pBitmapToSubtract->GetScanline( i );

                const size_t N = pSrcBitmap->GetWidth() * PixelFormatTraits<pixelFormat>::channelCount;

                for ( uint32_t j = 0; j < N; ++j )
                {
                    pSrcScanline[j] = std::max( 0, pSrcScanline[j] - pScanlineToSubtract[j] );
                }
            }
        } );
        this->_pDstBitmap = this->_pSrcBitmap;
    }
    
};

