#pragma once
#include "basetransform.h"
#include "../Geometry/size.h"
ACMB_NAMESPACE_BEGIN

class BinningTransform : public BaseTransform
{
protected:
    Size _bin;
    BinningTransform( std::shared_ptr<IBitmap> pSrcBitmap, Size bin );
public:
    static std::shared_ptr<BinningTransform> Create(std::shared_ptr<IBitmap> pSrcBitmap, Size bin);
};

ACMB_NAMESPACE_END