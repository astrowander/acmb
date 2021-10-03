#ifndef IMAGEPARAMS_H
#define IMAGEPARAMS_H

#include "enums.h"

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

#endif // IMAGEPARAMS_H
