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

    BitmapSubtractor( IBitmapPtr pSrcBitmap, IBitmapPtr pBitmapToSubtract );
    void Run() override;    
};
