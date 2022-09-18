#include "imageparams.h"
ACMB_NAMESPACE_BEGIN

ImageParams::ImageParams(uint32_t width, uint32_t height, PixelFormat pixelFormat)
: _width(width)
, _height(height)
, _pixelFormat(pixelFormat)
{

}

uint32_t ImageParams::GetWidth() const
{
    return _width;
}

uint32_t ImageParams::GetHeight() const
{
    return _height;
}

PixelFormat ImageParams::GetPixelFormat() const
{
    return _pixelFormat;
}

ACMB_NAMESPACE_END