#include "imageparams.h"

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

/*void ImageParams::SetWidth(uint32_t val)
{
    _width = val;
}*/

uint32_t ImageParams::GetHeight() const
{
    return _height;
}

/*void ImageParams::SetHeight(uint32_t val)
{
    _height = val;
}*/

PixelFormat ImageParams::GetPixelFormat() const
{
    return _pixelFormat;
}

/*void ImageParams::SetPixelFormat(PixelFormat val)
{
    _pixelFormat = val;
}
*/
