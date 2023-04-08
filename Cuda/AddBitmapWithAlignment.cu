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

__device__ TriangleTransformPair ChooseTriangle( const double2 p, const TriangleTransformPair* cell, const size_t cellSize )
{
    double minDist = DBL_MAX;
    TriangleTransformPair nearest;

    for ( size_t i = 0; i < cellSize; ++i )
    {
        const auto& pair = cell[i];
        if ( IsPointInside( pair.triangle, p ) )
            return pair;

        double dist = DistSq( p, GetCenter( pair.triangle ) );
        if ( dist < minDist )
        {
            nearest = pair;
            minDist = dist;
        }
    }

    return nearest;
}

__device__ void Transform( const TransAffine& transform, double* x, double* y )
{
    double tmp = *x;
    *x = tmp * transform.sx + *y * transform.shx + transform.tx;
    *y = tmp * transform.shy + *y * transform.sy + transform.ty;
}
 
template<typename ChannelType>
__device__ float GetInterpolatedChannel( float x, float y, uint32_t ch,
                                         ChannelType* data, uint32_t width, uint32_t height, uint32_t channelCount )
{
    if ( x < 0 || x > width - 1 )
        return 0.0f;

    if ( y < 0 || y > height - 1 )
        return 0.0f;    

    auto x0 = uint32_t( x + 0.5f );
    if ( x0 == width - 1 )
        x0 -= 2;
    else if ( x0 >= 1 )
        x0 -= 1;

    auto y0 = uint32_t( y + 0.5f );
    if ( y0 == height - 1 )
        y0 -= 2;
    else if ( y0 >= 1 )
        y0 -= 1;

    float yIn[3] =
    {
        QuadraticInterpolation( x - x0, data[( y0 * width + x0 ) * channelCount + ch], data[( y0 * width + x0 + 1 ) * channelCount + ch], data[( y0 * width + x0 + 2 ) * channelCount + ch] ),
        QuadraticInterpolation( x - x0, data[( ( y0 + 1 ) * width + x0 ) * channelCount + ch], data[( ( y0 + 1 ) * width + x0 + 1 ) * channelCount + ch], data[( ( y0 + 1 ) * width + x0 + 2 ) * channelCount + ch] ),
        QuadraticInterpolation( x - x0, data[( ( y0 + 2 ) * width + x0 ) * channelCount + ch], data[( ( y0 + 2 ) * width + x0 + 1 ) * channelCount + ch], data[( ( y0 + 2 ) * width + x0 + 2 ) * channelCount + ch] )
    };

    return Clamp<float>( QuadraticInterpolation( y - y0, yIn[0], yIn[1], yIn[2] ), 0, MaxValue<ChannelType>() );
}
template<typename ChannelType>
__global__ void kernel( const ChannelType* pixels, const uint32_t width, const uint32_t height, const uint32_t channelCount,
                        const TriangleTransformPair** grid, const uint32_t* cellSizes, const size_t gridWidth, const size_t gridHeight, const size_t gridPixelSize,
                        float* pMeans, float* pDevs, uint16_t* pCounts )
{
    const size_t size = width * height;    

    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if ( index >= size )
        return;

    uint32_t x = index % width;
    uint32_t y = index / width;

    double2 p{ double( x ), double( y ) };

    size_t hGridIndex = x / gridPixelSize;
    size_t vGridIndex = y / gridPixelSize;
    
    const size_t cellIndex = vGridIndex * gridWidth + hGridIndex;
    auto pair = ChooseTriangle( p, grid[cellIndex], cellSizes[cellIndex] );
    Transform( pair.transform, &p.x, &p.y );

    if ( p.x < 0 || p.x > width - 1 || p.y < 0 || p.y > height - 1 )
        return;
    
    for ( uint32_t ch = 0; ch < channelCount; ++ch )
    {
        const auto interpolatedChannel = GetInterpolatedChannel( float( p.x ), float( p.y ), ch, pixels, width, height, channelCount );
        const size_t channelIndex = y * width * channelCount + x * channelCount + ch;
        auto& mean = pMeans[channelIndex];
        auto& dev = pDevs[channelIndex];
        auto& n = pCounts[channelIndex];

        const auto sigma = sqrt( dev );
        const auto kappa = 3.0;

        if ( n <= 5 || fabs( mean - interpolatedChannel ) < kappa * sigma )
        {
            dev = n * ( dev + ( interpolatedChannel - mean ) * ( interpolatedChannel - mean ) / ( n + 1 ) ) / ( n + 1 );

            mean = Clamp( ( n * mean + interpolatedChannel ) / ( n + 1 ), 0.0f, float( MaxValue<ChannelType>() ) );
            ++n;
        }
    }    
}

template<typename ChannelType>
void AddBitmapWithAlignmentKernel( const ChannelType* pixels, const uint32_t width, const uint32_t height, const uint32_t channelCount,
                                   const TriangleTransformPair** grid, const uint32_t* cellSizes, const size_t gridWidth, const size_t gridHeight, const size_t gridPixelSize,
                                   float* pMeans, float* pDevs, uint16_t* pCounts )
{
    int maxThreadsPerBlock = 0;
    cudaDeviceGetAttribute( &maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0 );
    int numBlocks = ( int( size ) + maxThreadsPerBlock - 1 ) / maxThreadsPerBlock;
    kernel<ChannelType> << <numBlocks, maxThreadsPerBlock >> > ( pixels, width, height, channelCount,
                                                                 grid, cellSizes, gridWidth, gridHeight, gridPixelSize,
                                                                 pMeans, pDevs, pCounts );
}

ACMB_CUDA_NAMESPACE_END