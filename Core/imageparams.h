#pragma once

#include "enums.h"

ACMB_NAMESPACE_BEGIN

class ImageParams
{
protected:
    uint32_t _width;
    uint32_t _height;
    PixelFormat _pixelFormat;

public:
    ImageParams(uint32_t width = 0, uint32_t height = 0, PixelFormat pixelFormat = PixelFormat::Unknown);
    uint32_t GetWidth() const;
    uint32_t GetHeight() const;
    PixelFormat GetPixelFormat() const;
};

ACMB_NAMESPACE_END
