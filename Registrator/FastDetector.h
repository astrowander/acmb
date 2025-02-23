#pragma once
#include "./../Core/bitmap.h"

ACMB_NAMESPACE_BEGIN

std::vector<PointD> DetectFeatures( IBitmapPtr pBitmap, float threshold = 0.1f, int minChannel = 100 );

ACMB_NAMESPACE_END