#include "AddBitmapWithAlignment.h"
#include "AddBitmapWithAlignment.cuh"
#include "./../Core/bitmap.h"

ACMB_CUDA_NAMESPACE_BEGIN

void AddBitmapWithAlignmentHelper( std::shared_ptr<IBitmap> pBitmap, float* pMeans, float* pDevs, uint16_t* pCounts,
                                   const BaseStacker::Grid& grid, const size_t gridPixelSize, const size_t gridWidth, const size_t gridHeight )
{
    const size_t gridSize = gridWidth * gridHeight;
    Grid cudaGrid( gridSize );
    DynamicArrayU32 cellSizes( gridSize );

    std::vector<GridCell> cudaGridCells( gridSize );

    for ( size_t i = 0; i < gridWidth * gridHeight; ++i )
    {
        GridCell cell;
        cell.fromVector( grid[i] );
        cudaGridCells[i] = std::move( cell );
        cudaGrid.data()[i] = cudaGridCells[i].data();
        cellSizes.data()[i] = uint32_t( cudaGridCells[i].size() );
    }

    const uint32_t width = pBitmap->GetWidth();
    const uint32_t height = pBitmap->GetHeight();

    switch ( pBitmap->GetPixelFormat() )
    {
        case PixelFormat::Gray8:
        {
            DynamicArrayU8 pixels( width * height );
            pixels.fromVector( std::static_pointer_cast< Bitmap<PixelFormat::Gray8> >( pBitmap )->GetData() );
            AddBitmapWithAlignmentKernel<uint8_t>(pixels.data(), width, height, 1,
                                                   const_cast<const TriangleTransformPair**>( cudaGrid.data() ), cellSizes.data(), gridWidth, gridHeight, gridPixelSize,
                                                   pMeans, pDevs, pCounts );
            break;
        }
        case PixelFormat::Gray16:
        {
            DynamicArrayU16 pixels( width * height );
            pixels.fromVector( std::static_pointer_cast< Bitmap<PixelFormat::Gray16> >( pBitmap )->GetData() );
            AddBitmapWithAlignmentKernel<uint16_t>( pixels.data(), width, height, 1,
                                                   const_cast< const TriangleTransformPair** >( cudaGrid.data() ), cellSizes.data(), gridWidth, gridHeight, gridPixelSize,
                                                   pMeans, pDevs, pCounts );
            break;
        }
        case PixelFormat::RGB24:
        {
            DynamicArrayU8 pixels( width * height * 3 );
            pixels.fromVector( std::static_pointer_cast< Bitmap<PixelFormat::Gray8> >( pBitmap )->GetData() );
            AddBitmapWithAlignmentKernel<uint8_t>( pixels.data(), width, height, 3,
                                                    const_cast< const TriangleTransformPair** >( cudaGrid.data() ), cellSizes.data(), gridWidth, gridHeight, gridPixelSize,
                                                    pMeans, pDevs, pCounts );
            break;
        }
        case PixelFormat::RGB48:
        {
            DynamicArrayU16 pixels( width * height * 3 );
            pixels.fromVector( std::static_pointer_cast< Bitmap<PixelFormat::Gray16> >( pBitmap )->GetData() );
            AddBitmapWithAlignmentKernel<uint16_t>( pixels.data(), width, height, 3,
                                                   const_cast< const TriangleTransformPair** >( cudaGrid.data() ), cellSizes.data(), gridWidth, gridHeight, gridPixelSize,
                                                   pMeans, pDevs, pCounts );
            break;
        }
        case PixelFormat::Bayer16:
        {
            DynamicArrayU16 pixels( width * height );
            pixels.fromVector( std::static_pointer_cast< Bitmap<PixelFormat::Bayer16> >( pBitmap )->GetData() );
            AddBitmapWithAlignmentKernel<uint16_t>( pixels.data(), width, height, 1,
                                                    const_cast< const TriangleTransformPair** >( cudaGrid.data() ), cellSizes.data(), gridWidth, gridHeight, gridPixelSize,
                                                    pMeans, pDevs, pCounts );
            break;
        }
        default:
            break;

    }
}

ACMB_CUDA_NAMESPACE_END