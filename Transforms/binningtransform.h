#pragma once
#include "basetransform.h"
#include "../Geometry/size.h"
ACMB_NAMESPACE_BEGIN

class BinningTransform : public BaseTransform
{
public:
    using Settings = Size;

protected:
    Size _bin;
    BinningTransform( std::shared_ptr<IBitmap> pSrcBitmap, Size bin );
public:
    static std::shared_ptr<BinningTransform> Create(std::shared_ptr<IBitmap> pSrcBitmap, Size bin);
    static std::shared_ptr<BinningTransform> Create( PixelFormat pixelFormat, Size bin );
};

ACMB_NAMESPACE_END