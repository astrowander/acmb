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
    //void SetWidth(uint32_t val);

    uint32_t GetHeight() const;
    //void SetHeight(uint32_t val);

    PixelFormat GetPixelFormat() const;
    //void  SetPixelFormat(PixelFormat val);
};

#endif // IMAGEPARAMS_H
