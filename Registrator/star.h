#ifndef STAR_H
#define STAR_H
#include "Geometry/rect.h"

struct Star
{
    Rect rect;
    double luminance;
    uint32_t pixelCount;
};

#endif // STAR_H
