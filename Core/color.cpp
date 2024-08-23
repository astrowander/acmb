#include "color.h"

ACMB_NAMESPACE_BEGIN

std::shared_ptr<IColor> IColor::Create( PixelFormat pixelFormat, const std::array<uint32_t, 4>& channels )
{
    switch ( pixelFormat )
    {
    case PixelFormat::RGB24:
        return std::make_shared<Color<PixelFormat::RGB24>>( channels[0], channels[1], channels[2] );
    case PixelFormat::RGBA32:
        return std::make_shared<Color<PixelFormat::RGBA32>>( channels[0], channels[1], channels[2], channels[3] );
    case PixelFormat::RGB48:
        return std::make_shared<Color<PixelFormat::RGB48>>( channels[0], channels[1], channels[2] );
    case PixelFormat::RGBA64:
        return std::make_shared<Color<PixelFormat::RGBA64>>( channels[0], channels[1], channels[2], channels[3] );
    case PixelFormat::Gray8:
        return std::make_shared<Color<PixelFormat::Gray8>>( channels[0] );
    case PixelFormat::Gray16:
        return std::make_shared<Color<PixelFormat::Gray16>>( channels[0] );
    }

	throw std::runtime_error( "unsupported pixel format" );
}

std::shared_ptr<Color<PixelFormat::Gray8>> IColor::MakeGray8( uint8_t l )
{
    return std::make_shared<Color<PixelFormat::Gray8>>( l );
}

std::shared_ptr<Color<PixelFormat::Gray8>> IColor::MakeGray8( NamedColor32 color )
{
    return std::make_shared<Color<PixelFormat::Gray8>>( uint8_t( color ) );
}

std::shared_ptr<Color<PixelFormat::Gray16>> IColor::MakeGray16( uint16_t l )
{
    return std::make_shared<Color<PixelFormat::Gray16>>( l );
}

std::shared_ptr<Color<PixelFormat::Gray16>> IColor::MakeGray16( NamedColor64 color )
{
    return std::make_shared<Color<PixelFormat::Gray16>>( uint16_t( color ) );
}

std::shared_ptr<Color<PixelFormat::RGB24>> IColor::MakeRGB24( uint8_t r, uint8_t g, uint8_t b )
{
    return std::make_shared<Color<PixelFormat::RGB24>>( r, g, b );
}

std::shared_ptr<Color<PixelFormat::RGB24>> IColor::MakeRGB24( NamedColor32 color )
{
    uint32_t color32 = uint32_t( color );
    const uint8_t b = color32 & 0xff;
    const uint8_t g = ( color32 >> 8 ) & 0xff;
    const uint8_t r = ( color32 >> 16 ) & 0xff;
    return std::make_shared<Color<PixelFormat::RGB24>>( r, g, b );
}

std::shared_ptr<Color<PixelFormat::RGB48>> IColor::MakeRGB48( NamedColor64 color )
{
    uint64_t color32 = uint64_t( color );
    const uint16_t b = color32 & 0xffff;
    const uint16_t g = (color32 >> 16) & 0xffff;
    const uint16_t r = (color32 >> 32) & 0xffff;
    return std::make_shared<Color<PixelFormat::RGB48>>( r, g, b );
}

std::shared_ptr<Color<PixelFormat::RGB48>> IColor::MakeRGB48( uint16_t r, uint16_t g, uint16_t b )
{
    return std::make_shared<Color<PixelFormat::RGB48>>( r, g, b );
}

std::shared_ptr<Color<PixelFormat::RGBA32>> IColor::MakeRGBA32( uint8_t r, uint8_t g, uint8_t b, uint8_t a )
{
    return std::make_shared<Color<PixelFormat::RGBA32>>( r, g, b, a );
}

std::shared_ptr<Color<PixelFormat::RGBA32>> IColor::MakeRGBA32( NamedColor32 color )
{
    uint32_t color32 = uint32_t( color );
    const uint8_t b = color32 & 0xff;
    const uint8_t g = ( color32 >> 8 ) & 0xff;
    const uint8_t r = ( color32 >> 16 ) & 0xff;
    const uint8_t a = ( color32 >> 24 ) & 0xff;
    return std::make_shared<Color<PixelFormat::RGBA32>>( r, g, b, a );
}

std::shared_ptr<Color<PixelFormat::RGBA64>> IColor::MakeRGBA64( uint16_t r, uint16_t g, uint16_t b, uint16_t a )
{
    return std::make_shared<Color<PixelFormat::RGBA64>>( r, g, b, a );
}

std::shared_ptr<Color<PixelFormat::RGBA64>> IColor::MakeRGBA64( NamedColor64 color )
{
    uint64_t color32 = uint64_t( color );
    const uint16_t b = color32 & 0xffff;
    const uint16_t g = (color32 >> 16) & 0xffff;
    const uint16_t r = (color32 >> 32) & 0xffff;
    const uint16_t a = (color32 >> 48) & 0xffff;
    return std::make_shared<Color<PixelFormat::RGBA64>>( r, g, b, a );
}

ACMB_NAMESPACE_END