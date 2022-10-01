#pragma once
#include "../Core/bitmap.h"
#include <stdexcept>
#include "basetransform.h"
ACMB_NAMESPACE_BEGIN
/// <summary>
/// Converts an image from one pixel format to another one
/// </summary>
class Converter : public BaseTransform
{
public:
    using Settings = PixelFormat;

protected:
    Converter(IBitmapPtr pSrcBitmap);

public:
    /// Creates instance with source bitmap and destination pixel format
    static std::shared_ptr<Converter> Create(IBitmapPtr pSrcBitmap, PixelFormat dstPixelFormat);
    /// Creates instance with source pixel format and destination pixel format. Source bitmap must be set later
    static std::shared_ptr<Converter> Create( PixelFormat srcPixelFormat, PixelFormat dstPixelFormat );

    /// Converts an image from its own pixel format to another one and returns result
    static IBitmapPtr Convert(IBitmapPtr pSrcBitmap, PixelFormat dstPixelFormat);
};

ACMB_NAMESPACE_END


