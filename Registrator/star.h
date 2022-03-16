#ifndef STAR_H
#define STAR_H
#include "../Geometry/rect.h"

struct Star
{
    Rect rect;
    PointF center;
    double luminance = 0.0;
    uint32_t pixelCount = 0;  
};

#endif // STAR_H
