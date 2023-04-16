#pragma once
#include "./../Registrator/BaseStacker.h"

ACMB_NAMESPACE_BEGIN
class IBitmap;
ACMB_NAMESPACE_END

ACMB_CUDA_NAMESPACE_BEGIN

struct GridData;

template<typename ChannelType>
class AddBitmapWithAlignmentHelper
{
    std::shared_ptr<GridData> _pGridData;
public:
    AddBitmapWithAlignmentHelper();
    void Run( const ChannelType* pixels, const uint32_t width, const uint32_t height, const uint32_t channelCount,
              const BaseStacker::Grid& grid,
              float* pMeans, float* pDevs, uint16_t* pCounts );
};

using AddBitmapWithAlignmentHelperU8 = AddBitmapWithAlignmentHelper<uint8_t>;
using AddBitmapWithAlignmentHelperU16 = AddBitmapWithAlignmentHelper<uint16_t>;

ACMB_CUDA_NAMESPACE_END
