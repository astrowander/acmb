#include "AddBitmapWithAlignment.h"
#include "AddBitmapWithAlignment.cuh"
#include "./../Core/bitmap.h"

ACMB_CUDA_NAMESPACE_BEGIN
/*
inline double sqr(double d)
{
    return d * d;
}

inline double distSq( const double2& p1, const double2& p2 )
{
    return sqr( p2.x - p1.x ) + sqr( p2.y - p1.y );
}

inline double sign( const double2& p1, const double2& p2, const double2& p3 )
{
    return ( p1.x - p3.x ) * ( p2.y - p3.y ) - ( p2.x - p3.x ) * ( p1.y - p3.y );
}

bool isPointInside( const Triangle& triangle, const double2& p )
{
    auto d1 = sign( p, triangle[0], triangle[1] );
    auto d2 = sign( p, triangle[1], triangle[2] );
    auto d3 = sign( p, triangle[2], triangle[0] );

    bool hasNeg = ( d1 < 0 ) || ( d2 < 0 ) || ( d3 < 0 );
    bool hasPos = ( d1 > 0 ) || ( d2 > 0 ) || ( d3 > 0 );

    return !( hasNeg && hasPos );
}

double2 getCenter( const Triangle& triangle )
{
    return  { ( triangle[0].x + triangle[1].x + triangle[2].x ) / 3.0, ( triangle[0].y + triangle[1].y + triangle[2].y ) / 3.0 };
}

TriangleTransformPair chooseTriangle( const double2 p, const TriangleTransformPair* cell, const size_t cellSize )
{
    double minDist = DBL_MAX;
    TriangleTransformPair nearest;

    for ( size_t i = 0; i < cellSize; ++i )
    {
        const auto& pair = cell[i];
        if ( isPointInside( pair.triangle, p ) )
            return pair;

        double dist = distSq( p, getCenter( pair.triangle ) );
        if ( dist < minDist )
        {
            nearest = pair;
            minDist = dist;
        }
    }

    return nearest;
}

void transform( const TransAffine& transform, double* x, double* y )
{
    double tmp = *x;
    *x = tmp * transform.sx + *y * transform.shx + transform.tx;
    *y = tmp * transform.shy + *y * transform.sy + transform.ty;
}

template<typename ChannelType>
float GetInterpolatedChannel( float x, float y, uint32_t ch,
                                         const ChannelType* data, uint32_t width, uint32_t height, uint32_t channelCount )
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
*/
void AddBitmapWithAlignmentHelper( std::shared_ptr<IBitmap> pBitmap, float* pMeans, float* pDevs, uint16_t* pCounts,
                                   const BaseStacker::Grid& grid, const size_t gridPixelSize, const size_t gridWidth )
{
    const size_t gridSize = grid.size();
    
    DynamicArray<TriangleTransformPair> cudaGrid( grid );
    std::vector< TriangleTransformPair> cudaGridCopy;
    cudaGrid.toVector( cudaGridCopy );

    std::vector<uint32_t> cellOffsets( gridSize + 1 );

    for ( size_t i = 1; i <= gridSize; ++i )
    {
        cellOffsets[i] = cellOffsets[i - 1] + uint32_t( grid[i - 1].size() );
    }

    DynamicArrayU32 cudaCellOffsets( cellOffsets );

    const uint32_t width = pBitmap->GetWidth();
    const uint32_t height = pBitmap->GetHeight();

    switch ( pBitmap->GetPixelFormat() )
    {
        case PixelFormat::Gray8:
        {
            const DynamicArrayU8 pixels( std::static_pointer_cast< Bitmap<PixelFormat::Gray8> >( pBitmap )->GetData() );
            AddBitmapWithAlignmentKernel<uint8_t>( pixels.data(), width, height, 1,
                                                   cudaGrid.data(), cudaCellOffsets.data(), gridWidth, gridPixelSize,
                                                   pMeans, pDevs, pCounts );
            break;
        }
        case PixelFormat::Gray16:
        {
            const DynamicArrayU16 pixels( std::static_pointer_cast< Bitmap<PixelFormat::Gray16> >( pBitmap )->GetData() );
            AddBitmapWithAlignmentKernel<uint16_t>( pixels.data(), width, height, 1,
                                                    cudaGrid.data(), cudaCellOffsets.data(), gridWidth, gridPixelSize,
                                                    pMeans, pDevs, pCounts );
            break;
        }
        case PixelFormat::RGB24:
        {
            const DynamicArrayU8 pixels( std::static_pointer_cast< Bitmap<PixelFormat::RGB24> >( pBitmap )->GetData() );
            AddBitmapWithAlignmentKernel<uint8_t>( pixels.data(), width, height, 3,
                                                   cudaGrid.data(), cudaCellOffsets.data(), gridWidth, gridPixelSize,
                                                   pMeans, pDevs, pCounts );
            break;
        }
        case PixelFormat::RGB48:
        {
            DynamicArrayU16 pixels( std::static_pointer_cast< Bitmap<PixelFormat::RGB48> >( pBitmap )->GetData() );
            AddBitmapWithAlignmentKernel<uint16_t>( pixels.data(), width, height, 3,
                                                    cudaGrid.data(), cudaCellOffsets.data(), gridWidth, gridPixelSize,
                                                    pMeans, pDevs, pCounts );
            break;
        }
        case PixelFormat::Bayer16:
        {
            DynamicArrayU16 pixels( std::static_pointer_cast< Bitmap<PixelFormat::Bayer16> >( pBitmap )->GetData() );
            AddBitmapWithAlignmentKernel<uint16_t>( pixels.data(), width, height, 1,
                                                    cudaGrid.data(), cudaCellOffsets.data(), gridWidth, gridPixelSize,
                                                    pMeans, pDevs, pCounts );
            break;
        }
        default:
            break;

    }

    /*const auto& pixels = std::static_pointer_cast< Bitmap<PixelFormat::RGB48> >( pBitmap )->GetData();
    for ( size_t index = 0; index < width * height; ++index )
    {
        uint32_t x = index % width;
        uint32_t y = index / width;

        double2 p{ double( x ), double( y ) };

        size_t hGridIndex = x / gridPixelSize;
        size_t vGridIndex = y / gridPixelSize;

        const size_t cellIndex = vGridIndex * gridWidth + hGridIndex;
        const size_t cellSize = cellOffsets[cellIndex + 1] - cellOffsets[cellIndex];
        if ( cellSize > 0 )
        {
            auto pair = ChooseTriangle( p, &cudaGridCopy[cellOffsets[cellIndex]], cellSize );
            Transform( pair.transform, &p.x, &p.y );
        }

        if ( p.x < 0 || p.x > width - 1 || p.y < 0 || p.y > height - 1 )
            continue;

        for ( uint32_t ch = 0; ch < 3; ++ch )
        {
            const auto interpolatedChannel = GetInterpolatedChannel<uint16_t>( float( p.x ), float( p.y ), ch, &pixels[0], width, height, 3);
            const size_t channelIndex = y * width * 3 + x * 3 + ch;
            auto& mean = pMeans[channelIndex];
            auto& dev = pDevs[channelIndex];
            auto& n = pCounts[channelIndex];

            const auto sigma = sqrt( dev );
            const auto kappa = 3.0;

            if ( n <= 5 || fabs( mean - interpolatedChannel ) < kappa * sigma )
            {
                dev = n * ( dev + ( interpolatedChannel - mean ) * ( interpolatedChannel - mean ) / ( n + 1 ) ) / ( n + 1 );

                mean = Clamp( ( n * mean + interpolatedChannel ) / ( n + 1 ), 0.0f, float( MaxValue<uint16_t>() ) );
                ++n;
            }
        }
    }*/
}

ACMB_CUDA_NAMESPACE_END