#pragma once
#include "basetransform.h"
#include "./../Core/IParallel.h"

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
class BitmapSubtractor: public BaseBitmapSubtractor, public IParallel
{
    using ChannelType = typename PixelFormatTraits<pixelFormat>::ChannelType;

public:

    BitmapSubtractor( IBitmapPtr pSrcBitmap, IBitmapPtr pBitmapToSubtract )
    : BaseBitmapSubtractor(pSrcBitmap, pBitmapToSubtract)
    , IParallel(pSrcBitmap->GetHeight())
    { }

    void Job( uint32_t i ) override
    {
        auto pSrcBitmap = std::static_pointer_cast< Bitmap<pixelFormat> >( _pSrcBitmap );
        auto pBitmapToSubtract = std::static_pointer_cast< Bitmap<pixelFormat> >( _pBitmapToSubtract );
        //auto pDstBitmap = std::static_pointer_cast< Bitmap<pixelFormat> >( _pDstBitmap );
        auto pSrcScanline = pSrcBitmap->GetScanline( i );
        auto pScanlineToSubtract = pBitmapToSubtract->GetScanline( i );
        //auto pDstScanline = pDstBitmap->GetScanline( i );

        const size_t N = pSrcBitmap->GetWidth() * PixelFormatTraits<pixelFormat>::channelCount;

        for ( uint32_t j = 0; j < N ; ++j )
        {
            pSrcScanline[j] = std::max( 0, pSrcScanline[j] - pScanlineToSubtract[j] );            
        }
    }

    void Run() override
    {
        DoParallelJobs();
        this->_pDstBitmap = this->_pSrcBitmap;
    }

    
};

