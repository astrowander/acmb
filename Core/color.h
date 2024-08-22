#pragma once
#include "enums.h"
#include <array>

ACMB_NAMESPACE_BEGIN

template<PixelFormat pixelFormat>
struct Color;

struct IColor
{
    virtual PixelFormat GetPixelFormat() const = 0;
    virtual uint32_t GetChannel(uint32_t i) const = 0;
    virtual void SetChannel( uint32_t i, uint32_t value ) = 0;

    virtual ~IColor() = default;

    static std::shared_ptr<IColor> Create( PixelFormat pixelFormat, const std::array<uint32_t, 4>& channels );
    static std::shared_ptr<Color<PixelFormat::Gray8>> MakeGray8( uint8_t l );
    static std::shared_ptr<Color<PixelFormat::Gray8>> MakeGray8( NamedColor32 color );
    static std::shared_ptr<Color<PixelFormat::Gray16>> MakeGray16( uint16_t l );
    static std::shared_ptr<Color<PixelFormat::Gray16>> MakeGray16( NamedColor64 color );
    
    static std::shared_ptr<Color<PixelFormat::RGB24>> MakeRGB24( uint8_t r, uint8_t g, uint8_t b );
    static std::shared_ptr<Color<PixelFormat::RGB24>> MakeRGB24( NamedColor32 color );
    static std::shared_ptr<Color<PixelFormat::RGB48>> MakeRGB48( uint16_t r, uint16_t g, uint16_t b );
    static std::shared_ptr<Color<PixelFormat::RGB48>> MakeRGB48( NamedColor64 color );
};

using IColorPtr = std::shared_ptr<IColor>;

template<PixelFormat pixelFormat>
struct Color : IColor
{
    using ChannelType = typename PixelFormatTraits<pixelFormat>::ChannelType;
    using ColorType = typename PixelFormatTraits<pixelFormat>::IntColorType;
    using EnumColorType = typename PixelFormatTraits<pixelFormat>::EnumColorType;
    static constexpr uint32_t cChannelCount = PixelFormatTraits<pixelFormat>::channelCount;

    std::array<ChannelType, cChannelCount> channels;

    Color( ChannelType l ) requires (cChannelCount == 1)
    {
        channels[0] = l;
    }

    Color( ChannelType r, ChannelType g, ChannelType b ) requires (cChannelCount == 3)
    {
        channels[0] = r;
        channels[1] = g;
        channels[2] = b ;
    }

    Color( ChannelType r, ChannelType g, ChannelType b, ChannelType a = PixelFormatTraits<pixelFormat>::channelMax ) requires (cChannelCount == 4)
    {
        channels[0] = r;
        channels[1] = g;
        channels[2] = b;
        channels[3] = a;
    }

    Color( EnumColorType enumColor )
    {
        for ( uint32_t i = 0; i < cChannelCount; ++i )
        {
            auto channel = static_cast< ChannelType >(ColorType(enumColor) >> (BitsPerChannel( pixelFormat ) * i) & PixelFormatTraits<pixelFormat>::channelMax);
            channels[cChannelCount - i - 1] = channel;
        }
    }

    PixelFormat GetPixelFormat() const override
    {
        return pixelFormat;
    }

    uint32_t GetChannel(uint32_t i) const override
    {
        if (i >= cChannelCount)
            throw std::out_of_range("i");

        return channels[i];
    }

    void SetChannel( uint32_t i, uint32_t value ) override
    {
        if ( i >= cChannelCount )
            throw std::out_of_range( "i" );

        channels[i] = ChannelType( value );
    }
};

ACMB_NAMESPACE_END
