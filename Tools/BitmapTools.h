#pragma once
#include "../Core/bitmap.h"

ACMB_NAMESPACE_BEGIN

std::shared_ptr<Bitmap<PixelFormat::YUV24>> YUVBitmapFromPlanarData( std::vector<uint8_t>& yuv, uint32_t width, uint32_t height );

void PlanarDataFromYUVBitmap( std::shared_ptr<Bitmap<PixelFormat::YUV24>> pBitmap, std::vector<uint8_t>& yuv );


ACMB_NAMESPACE_END
