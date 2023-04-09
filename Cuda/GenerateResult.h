#pragma once
#include "./../Core/enums.h"
#include <cstdint>

ACMB_NAMESPACE_BEGIN

template<PixelFormat pixelFormat>
class Bitmap;

ACMB_NAMESPACE_END

ACMB_CUDA_NAMESPACE_BEGIN

template<PixelFormat pixelFormat>
std::shared_ptr<Bitmap<pixelFormat>> GeneratingResultHelper( float* pMeans, uint32_t width, uint32_t height );

ACMB_CUDA_NAMESPACE_END
