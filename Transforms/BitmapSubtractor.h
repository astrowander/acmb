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
        auto pDstBitmap = std::static_pointer_cast< Bitmap<pixelFormat> >( _pDstBitmap );
        auto pSrcScanline = pSrcBitmap->GetScanline( i );
        auto pScanlineToSubtract = pBitmapToSubtract->GetScanline( i );
        auto pDstScanline = pDstBitmap->GetScanline( i );

        for ( uint32_t j = 0; j < pDstBitmap->GetWidth() * PixelFormatTraits<pixelFormat>::channelCount; ++j )
        {
            pDstScanline[j] = std::max( 0, pSrcScanline[j] - pScanlineToSubtract[j] );            
        }
    }

    void Run() override
    {
        this->_pDstBitmap = std::make_shared<Bitmap<pixelFormat>>( _pSrcBitmap->GetWidth(), _pSrcBitmap->GetHeight() );
        DoParallelJobs();
    }   
};

