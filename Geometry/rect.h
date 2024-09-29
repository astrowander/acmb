#pragma once
#include "point.h"
#include "size.h"
ACMB_NAMESPACE_BEGIN
/// rectangle with given upper-left angle and width/height
template<typename T>
struct RectT
{
    T x = 0;
    T y = 0;
    T width = 0;
    T height = 0;
    /// return right border
    T GetRight() const
    {
        return x + width;
    }
    /// return bottom border
    T GetBottom() const
    {
        return y + height;
    }
    /// if given value exceeds current borders, sets right border to given value. Otherwise does nothing
    void ExpandRight(T right)
    {
        if (width < right - x + 1)
            width = right - x + 1;
    }
    
    /// if given value exceeds current borders, sets left border to given value. Otherwise does nothing
    void ExpandLeft(T left)
    {
        if (left < x)
        {
            x = left;
            width += x - left;
        }
    }
    /// if given value exceeds current borders, sets bottom border to given value. Otherwise does nothing
    void ExpandDown(T bottom)
    {
        if (bottom > y + height - 1)
        {
            height = bottom - y + 1;
        }
    }
    /// shifts rectangle
    void Translate(T tx, T ty)
    {
        x += tx;
        y += ty;
    }

    bool IsValid()
    {
        return width > 0 && height > 0;
    }

    inline bool operator==(const RectT& rhs) const;
    inline bool operator!=(const RectT& rhs) const;
    
    /// returns upper-left corner
    PointT<T> GetOrigin() const 
    {
        return PointT<T>{ x, y };
    }
    /// returns center point
    PointD GetCenter() const 
    {
        return { x + width / 2.0, y + height / 2.0 };
    }
    /// checks if the given point inside the rectangle
    bool IsPointInside(PointT<T> p) const
    {
        return (p.x >= x) && (p.x <= x + width) && (p.y >= y) && (p.y <= y + height);
    }
    /// checks if the rectangle overlaps with another one
    bool Overlaps(const RectT& other) const
    {
        return  (x < other.GetRight() && GetRight() > other.x && y < other.GetBottom() && GetBottom() > other.y);
    }
    /// prints rectangle parameters to a given stream
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
/// rect with integer coords
using Rect = RectT<int32_t>;
/// rect with fractional coords
using RectF = RectT<double>;

ACMB_NAMESPACE_END
