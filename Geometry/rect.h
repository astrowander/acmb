#pragma once
#include "point.h"
#include "size.h"
ACMB_NAMESPACE_BEGIN

template<typename T>
struct RectT
{
    T x = 0;
    T y = 0;
    T width = 0;
    T height = 0;

    T GetRight() const
    {
        return x + width;
    }

    T GetBottom() const
    {
        return y + height;
    }

    void ExpandRight(T right)
    {
        if (width < right - x + 1)
            width = right - x + 1;
    }

    void ExpandLeft(T left)
    {
        if (left < x)
        {
            x = left;
            width += x - left;
        }
    }

    void ExpandDown(T bottom)
    {
        if (bottom > y + height - 1)
        {
            height = bottom - y + 1;
        }
    }

    void Translate(T tx, T ty)
    {
        x += tx;
        y += ty;
    }

    inline bool operator==(const RectT& rhs) const;
    inline bool operator!=(const RectT& rhs) const;
    

    PointT<T> GetOrigin() const 
    {
        return PointT<T>{ x, y };
    }

    PointF GetCenter() const 
    {
        return { x + width / 2.0, y + height / 2.0 };
    }

    bool IsPointInside(PointT<T> p) const
    {
        return (p.x >= x) && (p.x <= x + width) && (p.y >= y) && (p.y <= y + height);
    }

    bool Overlaps(const RectT& other) const
    {
        return  (x < other.GetRight() && GetRight() > other.x && y < other.GetBottom() && GetBottom() > other.y);
    }

    template <typename U>
    friend std::ostream& operator <<(std::ostream& out, const RectT<U>& rect);    
};

template <typename T>
std::ostream& operator<<(std::ostream& out, const RectT<T>& rect)
{
    return out << rect.x << " " << rect.y << " " << rect.width << " " << rect.height;
}

template<typename T>
bool RectT<T>::operator==(const RectT<T>& rhs) const
{
    return (x == rhs.x) && (y == rhs.y) && (width == rhs.width) && (height == rhs.height);
}

template<typename T>
bool RectT<T>::operator!=(const RectT<T>& rhs) const
{
    return !(*this == rhs);
}

using Rect = RectT<int32_t>;
using RectF = RectT<double>;

ACMB_NAMESPACE_END
