#include "AddBitmapWithAlignment.h"
#include "AddBitmapWithAlignment.cuh"
#include "./../Core/bitmap.h"
#include "./../Registrator/StackEngineConstants.h"

ACMB_CUDA_NAMESPACE_BEGIN

struct GridData
{
    DynamicArray<TriangleTransformPair> grid;
    DynamicArrayU32 gridOffsets;
};

template<typename ChannelType>
AddBitmapWithAlignmentHelper<ChannelType>::AddBitmapWithAlignmentHelper()
{
    _pGridData = std::make_shared<GridData>();
}

template<typename ChannelType>
void AddBitmapWithAlignmentHelper<ChannelType>::Run( const ChannelType* pixels, const uint32_t width, const uint32_t height, const uint32_t channelCount,
                                                const BaseStacker::Grid& grid,
                                                float* pMeans, float* pDevs, uint16_t* pCounts )
{
    _pGridData->grid.fromVectors( grid );
    const size_t gridSize = grid.size();
    std::vector<uint32_t> cellOffsets( gridSize + 1 );
    for ( size_t i = 1; i <= gridSize; ++i )
        cellOffsets[i] = cellOffsets[i - 1] + uint32_t( grid[i - 1].size() );

    _pGridData->gridOffsets.fromVector( cellOffsets );

    const size_t gridWidth  = width / cGridPixelSize + ( ( width % cGridPixelSize ) ? 1 : 0 );
    AddBitmapWithAlignmentKernel<ChannelType>( pixels, width, height, channelCount,
                                               _pGridData->grid.data(), _pGridData->gridOffsets.data(), gridWidth, cGridPixelSize,
                                               pMeans, pDevs, pCounts );
}

template class AddBitmapWithAlignmentHelper<uint8_t>;
template class AddBitmapWithAlignmentHelper<uint16_t>;

ACMB_CUDA_NAMESPACE_END