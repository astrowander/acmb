#ifndef MATHTOOLS_H
#define MATHTOOLS_H
#include <algorithm>
#include <tuple>

float QuadraticInterpolation(float t, float t0, float t1, float t2);

template<typename T>
T FastRound(double x)
{
    return static_cast<T>(x > 0 ? x + 0.5 : x - 0.5);
}

static constexpr uint32_t tileSize = 600;

constexpr std::tuple<uint32_t, uint32_t> GetTileCounts( uint32_t width, uint32_t height )
{
    return { std::max<uint32_t>( 1u, width / tileSize ), std::max<uint32_t>( 1u, height / tileSize ) };
}

#endif
