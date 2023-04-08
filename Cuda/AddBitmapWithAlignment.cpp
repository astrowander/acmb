#include "AddBitmapWithAlignment.h"
#include "AddBitmapWithAlignment.cuh"
#include "./../Core/bitmap.h"
#include "./../Registrator/BaseStacker.h"

ACMB_CUDA_NAMESPACE_BEGIN

void AddBitmapWithAlignmentHelper( std::shared_ptr<IBitmap> pBitmap, float* pMeans, float* pDevs, uint16_t* pCounts,
                                   const BaseStacker::Grid& grid, const size_t gridPixelSize, const size_t gridWidth, const size_t gridHeight )
{
    const size_t gridSize = gridWidth * gridHeight;
    Grid cudaGrid( gridSize );
    std::vector<GridCell> cudaGridCells( gridSize );

    for ( size_t i = 0; i < gridWidth * gridHeight; ++i )
    {
        GridCell cell;
        cell.fromVector( grid[i] );
        cudaGridCells[i] = std::move( cell );
        cudaGrid.data()[i] = cudaGridCells[i].data();
    }
}

ACMB_CUDA_NAMESPACE_END