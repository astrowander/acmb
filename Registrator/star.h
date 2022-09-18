#pragma once
#include "../Geometry/rect.h"
ACMB_NAMESPACE_BEGIN

struct Star
{
    Rect rect;
    PointF center;
    double luminance = 0.0;
    uint32_t pixelCount = 0;  
};

ACMB_NAMESPACE_END
