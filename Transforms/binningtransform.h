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
    BinningTransform( std::shared_ptr<IBitmap> pSrcBitmap, Size bin );
    void Run() override;

private:
    void ProcessPixel( ChannelType* pSrcPixel, ChannelType* pDstPixel );
};

