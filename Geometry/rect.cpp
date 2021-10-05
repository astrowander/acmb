#include "rect.h"


void Rect::ExpandRight(int32_t right)
{
    if (width < static_cast<uint32_t>(right - x + 1))
        width = right - x + 1;
}

void Rect::ExpandLeft(int32_t left)
{
    if (left < x)
    {
        x = left;
        width += x - left;
    }
}

void Rect::ExpandDown(int32_t bottom)
{
    if (bottom > y + static_cast<int32_t>(height) - 1)
    {
        height = bottom - y + 1;
    }
}

void Rect::Translate(int32_t tx, int32_t ty)
{
    x += tx;
    y += ty;
}

bool Rect::operator==(const Rect &rhs)
{
    return (x == rhs.x) && (y == rhs.y) && (width = rhs.width) && (height == rhs.height);
}

bool Rect::operator!=(const Rect &lhs)
{
    return !(*this == lhs);
}

double PointF::Distance(const PointF &rhs)
{
    return sqrt((rhs.x - x) * (rhs.x - x) + (rhs.y - y) * (rhs.y - y));
}
