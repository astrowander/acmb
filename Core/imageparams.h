#pragma once

#include "enums.h"

ACMB_NAMESPACE_BEGIN

/// represents parameters of image: width, weight and pixel format
class ImageParams
{
protected:
    uint32_t _width;
    uint32_t _height;
    PixelFormat _pixelFormat;

public:
    ImageParams(uint32_t width = 0, uint32_t height = 0, PixelFormat pixelFormat = PixelFormat::Unspecified);
    virtual ~ImageParams() = default;

    uint32_t GetWidth() const;
    uint32_t GetHeight() const;
    PixelFormat GetPixelFormat() const;
};

ACMB_NAMESPACE_END
