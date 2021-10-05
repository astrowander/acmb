#ifndef RECT_H
#define RECT_H
#include <cstdint>
#include <math.h>

struct Point
{
    int32_t x = 0;
    int32_t y = 0;
};

struct PointF
{
    double x = 0;
    double y = 0;

    double Distance(const PointF& rhs);
};

struct Rect
{
    int32_t x = 0;
    int32_t y = 0;
    uint32_t width = 0;
    uint32_t height = 0;

    void ExpandRight(int32_t right);
    void ExpandLeft(int32_t left);
    void ExpandDown(int32_t bottom);

    void Translate(int32_t x, int32_t y);

    bool operator==(const Rect& rhs);

    bool operator!=(const Rect& lhs);

    Point GetOrigin()
    {
        return Point {x, y};
    }

    PointF GetCenter()
    {
        return {x + width / 2.0, y + height / 2.0 };
    }
};

#endif // RECT_H
