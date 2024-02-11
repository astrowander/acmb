#pragma once
#include "macros.h"
#include <cstdint>
#include <type_traits>
#include <iostream>
#include <limits>

ACMB_NAMESPACE_BEGIN

/// represents pixel format of an image
enum class PixelFormat : uint32_t
{
    Unspecified = 0x0,
    RGB24 = 0x010103,
    RGBA32 = 0x010104,
    RGB48 = 0x020103,
    RGBA64 = 0x020104,

    Gray8 = 0x010201,
   // Agray16 = 0x010202,
    Gray16 = 0x020201,
   // Agray32 = 0x020202
   Bayer16 = 0x020301
};
/// prints pixel format to stream
std::ostream& operator<<(std::ostream& out, const PixelFormat& pixelFormat);
/// represents color space of an image
enum class ColorSpace : uint32_t
{
    RGB = 1,
    Gray = 2,
    Bayer = 3
};
/// returns color space of given pixel format
inline constexpr ColorSpace GetColorSpace( PixelFormat pixelFormat )
{
    return static_cast< ColorSpace >( ( static_cast< uint32_t >( pixelFormat ) >> 8 ) & 0xFF );
};
/// returns number of channels in given pixel format
inline constexpr uint32_t ChannelCount( PixelFormat pixelFormat )
{
    return ( uint32_t ) pixelFormat & 0xFF;
}
/// returns number of bytes in single channel in given pixel format
inline constexpr uint32_t BytesPerChannel( PixelFormat pixelFormat )
{
    return ( ( uint32_t ) pixelFormat >> 16 );
}
/// returns number of bytes in pixel of given format
inline constexpr uint32_t BytesPerPixel( PixelFormat pixelFormat )
{
    return BytesPerChannel( pixelFormat ) * ChannelCount( pixelFormat );
}
/// returns number of bits in single channel in given pixel format
inline constexpr uint32_t BitsPerChannel( PixelFormat pixelFormat )
{
    return BytesPerChannel( pixelFormat ) * 8;
}
/// returns number of bits in pixel of given format
inline constexpr uint32_t BitsPerPixel( PixelFormat pixelFormat )
{
    return BytesPerPixel( pixelFormat ) * 8;
}
///predefined rgb colors
enum class ARGB32Color : uint32_t
{
    Black       = 0xFF'00'00'00,
    Red         = 0xFF'FF'00'00,
    Green       = 0xFF'00'FF'00,
    Blue        = 0xFF'00'00'FF,
    White       = 0xFF'FF'FF'FF,
    Gray        = 0xFF'7F'7F'7F,
    Azure       = 0xFF'00'7F'FF,
    Transparent = 0x00'00'00'00
};
///predefined extended rgb colors
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
///creates rgb color from given channel values
inline constexpr uint32_t MakeRGB24( uint8_t r, uint8_t g, uint8_t b )
{
    return uint32_t{ b } + ( uint32_t{ g } << 8 ) + ( uint32_t{ r } << 16 );
}
///creates extended rgb color from given channel values
inline constexpr uint64_t MakeRGB48( uint16_t r, uint16_t g, uint16_t b )
{
    return uint64_t{ b } + ( uint64_t{ g } << 16 ) + ( uint64_t{ r } << 32 );
}
///aliases for channels
enum class Channel : uint32_t
{
    L = 0,
    R = 0,
    G = 1,
    B = 2,
    A = 3
};

///traits of given pixel format 
template<PixelFormat pixelFormat>
struct PixelFormatTraits
{
    /// alias for channel type (uint8_t/uint16_t)
    using ChannelType = typename std::conditional_t<((uint32_t)pixelFormat >> 16) == 1, uint8_t, uint16_t>;
    /// alias for color type (uint32_t/uint64_t)
    using ColorType = typename std::conditional_t<((uint32_t)pixelFormat >> 16) == 1, uint32_t, uint64_t>;
    /// alias for predefined enum color type (ARGB32Color/ARGB64Color)
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
/// creates pixel format from number of bits per channel and number of channels
PixelFormat ConstructPixelFormat( uint16_t bitsPerChannel, uint16_t channelsPerPixel );

constexpr bool ArePixelFormatsCompatible( PixelFormat f1, PixelFormat f2 )
{
    if ( f1 == f2 ||
         (f1 == PixelFormat::Gray16 && f2 == PixelFormat::Bayer16) ||
         (f1 == PixelFormat::Bayer16 && f2 == PixelFormat::Gray16) )
    {
        return true;
    }

    return false;
}

/// mode of ppm file format: text or binary
enum class PpmMode
{
    Text,
    Binary
};

enum class StackMode
{
    Light,
    LightNoAlign,
    DarkOrFlat,
    StarTrails
};

ACMB_NAMESPACE_END
