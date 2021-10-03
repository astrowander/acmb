#ifndef PIXELFORMATS_H
#define PIXELFORMATS_H

#include <cstdint>

enum class PixelFormat : uint32_t
{
    Unknown = 0x0,
    RGB24 = 0x010103,
    //ARGB32 = 0x010104,
    RGB48 = 0x020103,
    //ARGB64 = 0x020104,

    Gray8 = 0x010201,
   // Agray16 = 0x010202,
    Gray16 = 0x020201,
   // Agray32 = 0x020202
};

enum class ColorSpace : uint32_t
{
    RGB = 1,
    Gray = 2
};

constexpr ColorSpace GetColorSpace(PixelFormat pixelFormat)
{
    return static_cast<ColorSpace>((static_cast<uint32_t>(pixelFormat) >> 8) & 0xFF);
}

constexpr uint32_t ChannelCount(PixelFormat pixelFormat)
{
    return (uint32_t)pixelFormat & 0xFF;
}

constexpr uint32_t BytesPerChannel(PixelFormat pixelFormat)
{
    return ((uint32_t)pixelFormat >> 16);
}

constexpr uint32_t BytesPerPixel(PixelFormat pixelFormat)
{
    return BytesPerChannel(pixelFormat) * ChannelCount(pixelFormat);
}

constexpr uint32_t BitsPerChannel(PixelFormat pixelFormat)
{
    return BytesPerChannel(pixelFormat) * 8;
}

constexpr uint32_t BitsPerPixel(PixelFormat pixelFormat)
{
    return BytesPerPixel(pixelFormat) * 8;
}

enum class ARGB32Color : uint32_t
{
    Black       = 0xFF000000,
    Red         = 0xFFFF0000,
    Green       = 0xFF00FF00,
    Blue        = 0xFF0000FF,
    White       = 0xFFFFFFFF,
    Gray        = 0xFF7F7F7F,
    Transparent = 0x00000000
};

enum class ARGB64Color : uint64_t
{
    Black       = 0xFFFF'0000'0000'0000,
    Red         = 0xFFFF'FFFF'0000'0000,
    Green       = 0xFFFF'0000'FFFF'0000,
    Blue        = 0xFFFF'0000'0000'FFFF,
    White       = 0xFFFF'FFFF'FFFF'FFFF,
    Gray        = 0xFFFF'7FFF'7FFF'7FFF,
    Transparent = 0x0000'0000'0000'0000
};

enum class Channel : uint32_t
{
    L = 0,
    R = 0,
    G = 1,
    B = 2
};

enum class PpmMode
{
    Text,
    Binary
};

#endif // PIXELFORMATS_H
