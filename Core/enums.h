#ifndef PIXELFORMATS_H
#define PIXELFORMATS_H

#include <cstdint>
#include <type_traits>
#include <iostream>
#include <limits>

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

std::ostream& operator<<(std::ostream& out, const PixelFormat& pixelFormat);

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
    Azure       = 0xFF007FFF,
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
    Azure       = 0xFFFF'0000'7FFF,FFFF,
    Transparent = 0x0000'0000'0000'0000
};

constexpr uint32_t MakeRGB24( uint8_t r, uint8_t g, uint8_t b )
{
    return uint32_t{ b } + ( uint32_t{ g } << 8 ) + ( uint32_t{ r } << 16 );
}

constexpr uint64_t MakeRGB48( uint16_t r, uint16_t g, uint16_t b )
{
    return uint64_t{ b } + ( uint64_t{ g } << 16 ) + ( uint64_t{ r } << 32 );
}

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

template<PixelFormat pixelFormat>
struct PixelFormatTraits
{
    using ChannelType = typename std::conditional_t<((uint32_t)pixelFormat >> 16) == 1, uint8_t, uint16_t>;
    using ColorType = typename std::conditional_t<((uint32_t)pixelFormat >> 16) == 1, uint32_t, uint64_t>;
    using EnumColorType = typename std::conditional_t<((uint32_t)pixelFormat >> 16) == 1, ARGB32Color, ARGB64Color>;
    static constexpr auto colorSpace = GetColorSpace(pixelFormat);
    static constexpr auto channelCount = ChannelCount(pixelFormat);
    static constexpr auto bytesPerChannel = BytesPerChannel(pixelFormat);
    static constexpr auto bitsPerChannel = BitsPerChannel(pixelFormat);
    static constexpr auto bytesPerPixel = BytesPerPixel(pixelFormat);
    static constexpr auto bitsPerPixel = BitsPerPixel(pixelFormat);
    static constexpr auto channelMax = std::numeric_limits<ChannelType>::max();
    PixelFormatTraits() = delete;
};

#define CALL_HELPER(Helper, pBitmap) \
    switch (pBitmap->GetPixelFormat()) \
    { \
        case PixelFormat::Gray8: \
            Helper<PixelFormat::Gray8>::Run(*this, std::static_pointer_cast<Bitmap<PixelFormat::Gray8>>(pBitmap)); \
            break; \
        case PixelFormat::Gray16:\
             Helper<PixelFormat::Gray16>::Run(*this, std::static_pointer_cast<Bitmap<PixelFormat::Gray16>>(pBitmap)); \
            break; \
        case PixelFormat::RGB24:\
             Helper<PixelFormat::RGB24>::Run(*this, std::static_pointer_cast<Bitmap<PixelFormat::RGB24>>(pBitmap)); \
            break; \
        case PixelFormat::RGB48:\
              Helper<PixelFormat::RGB48>::Run(*this, std::static_pointer_cast<Bitmap<PixelFormat::RGB48>>(pBitmap)); \
            break; \
        default:\
            throw std::runtime_error("pixel format should be known");}

#endif // PIXELFORMATS_H
