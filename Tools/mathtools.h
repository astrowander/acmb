#pragma once
#include "../Core/macros.h"
#include "../Geometry/point.h"

#include <algorithm>
#include <tuple>
#include <array>
#include <cmath>
#include <limits>
#include <span>
#include <cstdint>

ACMB_NAMESPACE_BEGIN
/// Builds a parabola with three points: (0;t0) (1;t1) (2;t2) and find its value with arg = t
float QuadraticInterpolation(float t, float t0, float t1, float t2);
/// Builds a parabola with three points: (x0;y0) (x1;y1) (x2;y2) and find its value with arg = x
float ArbitraryQuadraticInterpolation( float x, float x0, float y0, float x1, float y1, float x2, float y2 );

/// rounds given value and cast it to given type
template<typename T>
constexpr T FastRound(double x)
{
    return static_cast<T>(x > 0 ? x + 0.5 : x - 0.5);
}

static constexpr uint32_t tileSize = 600;
/// returns nuber of tiles in the  image of specified size
constexpr std::tuple<uint32_t, uint32_t> GetTileCounts( uint32_t width, uint32_t height )
{
    return { std::max<uint32_t>( 1u, width / tileSize ), std::max<uint32_t>( 1u, height / tileSize ) };
}
/// converts RGB color to HSL color space
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

/// converts HSL color to RGB color space
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
/// calculates value of Gaussian in the point x 
/// xmax and ymax are coords of maximum point
/// sigma is standard deviation
float NormalDist( float x, float xmax, float ymax, float sigma );

template<typename T>
PointT<T> CubicBezier( PointT<T> p0, PointT<T> p1, PointT<T> p2, PointT<T> p3, T t )
{
    const T oneMinusT = T( 1 ) - t;
    return oneMinusT * oneMinusT * oneMinusT * p0
        + 3 * t * oneMinusT * oneMinusT * p1
        + 3 * t * t * oneMinusT * p2
        + t * t * t * p3;
}

ACMB_NAMESPACE_END
