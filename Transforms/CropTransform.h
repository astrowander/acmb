#pragma once
#include "basetransform.h"
#include "../Geometry/rect.h"
#include "./../Core/color.h"

ACMB_NAMESPACE_BEGIN
/// <summary>
/// Crops image to given rectangle
/// </summary>
class CropTransform : public BaseTransform
{
public:
    using Settings = Rect;

protected:
    Rect _dstRect;
    CropTransform( std::shared_ptr<IBitmap> pSrcBitmap, Rect dstRect );
public:
    /// Creates an instance with given bitmap and destination rectangle
    static std::shared_ptr<CropTransform> Create( IBitmapPtr pSrcBitmap, Rect dstRect );
    /// Creates an instance with given pixel format (need to set source bitmap later) and destination rectangle
    static std::shared_ptr<CropTransform> Create( PixelFormat pixelFormat, Rect dstRect );
    /// Crops image to the arbitrary size
    static IBitmapPtr Crop( IBitmapPtr pSrcBitmap, Rect dstRect );

    static IBitmapPtr CropAndFill( IBitmapPtr pSrcBitmap, Rect dstRect, IColorPtr pColor );

    /// returns size of destination image
    virtual void CalcParams( std::shared_ptr<ImageParams> pParams ) override;
};

ACMB_NAMESPACE_END