#pragma once
#include "basetransform.h"
#include "../Geometry/size.h"
ACMB_NAMESPACE_BEGIN
/// <summary>
/// Merges each region of given size (bin) in the source image to the single pixel in the destination
/// </summary>
class BinningTransform : public BaseTransform
{
public:
    using Settings = Size;

protected:
    Size _bin;
    BinningTransform( std::shared_ptr<IBitmap> pSrcBitmap, Size bin );
public:
    /// Creates an instance with given bitmap and bin size
    static std::shared_ptr<BinningTransform> Create(std::shared_ptr<IBitmap> pSrcBitmap, Size bin);
    /// Creates an instance with given pixel format (need to set source bitmap later) and bin size
    static std::shared_ptr<BinningTransform> Create( PixelFormat pixelFormat, Size bin );

    /// returns size of destination image
    virtual void CalcParams( std::shared_ptr<ImageParams> pParams ) override;
};

ACMB_NAMESPACE_END