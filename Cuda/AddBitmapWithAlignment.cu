#include "AddBitmapWithAlignment.cuh"

ACMB_CUDA_NAMESPACE_BEGIN

__device__ inline double Sqr( double d )
{
    return d * d;
}

__device__ inline double DistSq( const double2& p1, const double2& p2 )
{
    return Sqr( p2.x - p1.x ) + Sqr( p2.y - p1.y );
}

__device__ double Sign( const double2& p1, const double2& p2, const double2& p3 )
{
    return ( p1.x - p3.x ) * ( p2.y - p3.y ) - ( p2.x - p3.x ) * ( p1.y - p3.y );
}

__device__ bool IsPointInside( const Triangle& triangle, const double2& p )
{
    auto d1 = Sign( p, triangle[0], triangle[1] );
    auto d2 = Sign( p, triangle[1], triangle[2] );
    auto d3 = Sign( p, triangle[2], triangle[0] );

    bool hasNeg = ( d1 < 0 ) || ( d2 < 0 ) || ( d3 < 0 );
    bool hasPos = ( d1 > 0 ) || ( d2 > 0 ) || ( d3 > 0 );

    return !( hasNeg && hasPos );
}

__device__ double2 GetCenter( const Triangle& triangle )
{
    return  { ( triangle[0].x + triangle[1].x + triangle[2].x ) / 3.0, ( triangle[0].y + triangle[1].y + triangle[2].y ) / 3.0 };
}

__device__ void ChooseTriangle( const double2 p, TriangleTransformPair& lastPair, const TriangleTransformPair* cell, const size_t cellSize )
{
    if ( IsPointInside( lastPair.triangle, p ) )
        return;

    double minDist = DBL_MAX;
    TriangleTransformPair nearest;

    for ( size_t i = 0; i < cellSize; ++i )
    {
        const auto& pair = cell[i];
        if ( IsPointInside( pair.triangle, p ) )
        {
            lastPair = pair;
            return;
        }

        double dist = DistSq( p, GetCenter( pair.triangle ) );
        if ( dist < minDist )
        {
            nearest = pair;
            minDist = dist;
        }
    }

    lastPair = nearest;
}

ACMB_CUDA_NAMESPACE_END