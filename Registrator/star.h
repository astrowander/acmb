#pragma once
#include "../Geometry/rect.h"
ACMB_NAMESPACE_BEGIN



struct Star
{
    struct MomentMatrix
    {
        double xx = 0.0;
        double yy = 0.0;
        double xy = 0.0;

        auto operator<=>(const MomentMatrix& rhs) const = default;
    };

    Rect rect;
    PointD center;
    double luminance = 0.0;

    MomentMatrix m;

    uint32_t pixelCount = 0;
    bool isClipped = false;

    bool operator==(const Star& rhs) const
    {
        // at least one pixel should be shared between the stars
        return center.SquaredDistance(rhs.center) < 1;
    }
};

struct StarHash
{
    std::size_t operator()(const Star& star) const
    {
        std::size_t h1 = std::hash<int>{}(int(star.center.x + 0.5));
        std::size_t h2 = std::hash<int>{}(int(star.center.y + 0.5));
        return h1 ^ (h2 << 1);
    }
};

ACMB_NAMESPACE_END
