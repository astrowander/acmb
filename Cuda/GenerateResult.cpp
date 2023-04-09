#include "GenerateResult.h"
#include "GenerateResult.cuh"
#include "CudaBasic.h"
#include "./../Core/bitmap.h"

ACMB_CUDA_NAMESPACE_BEGIN

template<PixelFormat pixelFormat>
std::shared_ptr<Bitmap<pixelFormat>> GeneratingResultHelper( const float* pMeans, uint32_t width, uint32_t height )
{
    auto pRes = std::make_shared<Bitmap<pixelFormat>>( width, height );
    using ChannelType = typename PixelFormatTraits<pixelFormat>::ChannelType;
    const size_t size = width * height * PixelFormatTraits<pixelFormat>::channelCount;
    DynamicArray<ChannelType> cudaBitmap( size );
    GeneratingResultKernel( pMeans, cudaBitmap.data(), size );
    cudaBitmap.toVector( pRes->GetData() );
    return pRes;
}

template std::shared_ptr<Bitmap<PixelFormat::Gray8>> GeneratingResultHelper<PixelFormat::Gray8>( const float* pMeans, uint32_t width, uint32_t height );
template std::shared_ptr<Bitmap<PixelFormat::Gray16>> GeneratingResultHelper<PixelFormat::Gray16>( const float* pMeans, uint32_t width, uint32_t height );
template std::shared_ptr<Bitmap<PixelFormat::RGB24>> GeneratingResultHelper<PixelFormat::RGB24>( const float* pMeans, uint32_t width, uint32_t height );
template std::shared_ptr<Bitmap<PixelFormat::RGB48>> GeneratingResultHelper<PixelFormat::RGB48>( const float* pMeans, uint32_t width, uint32_t height );
template std::shared_ptr<Bitmap<PixelFormat::Bayer16>> GeneratingResultHelper<PixelFormat::Bayer16>( const float* pMeans, uint32_t width, uint32_t height );

ACMB_CUDA_NAMESPACE_END
