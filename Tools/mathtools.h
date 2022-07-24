#ifndef MATHTOOLS_H
#define MATHTOOLS_H
#include <algorithm>
#include <tuple>
#include <array>
#include <cmath>
#include <limits>
#include <span>

#undef max
#undef min

float QuadraticInterpolation(float t, float t0, float t1, float t2);

template<typename T>
constexpr T FastRound(double x)
{
    return static_cast<T>(x > 0 ? x + 0.5 : x - 0.5);
}

static constexpr uint32_t tileSize = 600;

constexpr std::tuple<uint32_t, uint32_t> GetTileCounts( uint32_t width, uint32_t height )
{
    return { std::max<uint32_t>( 1u, width / tileSize ), std::max<uint32_t>( 1u, height / tileSize ) };
}

template <typename ChannelType>
std::array<float, 3> RgbToHsl( const std::span<ChannelType, 3>& rgb )
{
    constexpr float channelMax = std::numeric_limits<ChannelType>::max();
    const float r = rgb[0] / channelMax;
    const float g = rgb[1] / channelMax;
    const float b = rgb[2] / channelMax;

    const auto min = std::min<float>( { r, g, b } );
    const auto max = std::max<float>( { r, g, b } );
    const auto chroma = max - min;

    std::array<float, 3> hsl = {};

    if ( chroma > std::numeric_limits<float>::epsilon() )
    {
        if ( max == r )
            hsl[0] = ( ( g - b ) / chroma + ( ( g < b ) ? 6 : 0 ) ) * 60.0f;
        else if ( max == g )
            hsl[0] = ( ( b - r ) / chroma + 2 ) * 60.0f;
        else
            hsl[0] = (( r - g ) / chroma + 4) * 60.0f;
    }

    hsl[2] = ( min + max ) / 2;
    hsl[1] = 0;

    if ( hsl[2] > 0 && hsl[2] < 1 )
    {
        hsl[1] = chroma / ( 1 - std::fabs( 2 * hsl[2] - 1 ) );
    }

    return hsl;
}

template <typename ChannelType>
void HslToRgb( const std::array<float, 3>& hsl, std::span<ChannelType, 3> & rgb )
{
    const auto f = [&hsl] ( int n )
    {
        float k = ( n + hsl[0] / 30.0f );
        k -= int( k ) / 12 * 12;
        auto res = hsl[2] - hsl[1] * std::min( hsl[2], 1 - hsl[2] ) * std::max( -1.0f, std::min( { k - 3, 9 - k, 1.0f } ) );
        return res;
    };

    constexpr float channelMax = std::numeric_limits<ChannelType>::max();
    rgb[0] = ChannelType( std::clamp( f( 0 ) * channelMax, 0.0f, channelMax ) + 0.5f );
    rgb[1] = ChannelType( std::clamp( f( 8 ) * channelMax, 0.0f, channelMax ) + 0.5f );
    rgb[2] = ChannelType( std::clamp( f( 4 ) * channelMax, 0.0f, channelMax ) + 0.5f );
}

float normalDist( float x, float xmax, float ymax, float sigma );

#endif
