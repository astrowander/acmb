#pragma once
#include "./../Core/macros.h"
#include <cstdint>
#include <iostream>
#include <math.h>

ACMB_NAMESPACE_BEGIN

/// represents 2D point
template<typename T>
struct PointT
{
	T x = 0;
	T y = 0;

	constexpr PointT operator+=(const PointT& rhs)
    {
		x += rhs.x; y += rhs.y;
        return *this;
    }

	constexpr PointT operator+( const PointT& rhs ) const
	{
		PointT ret = *this;
        return ret += rhs;
	}

    constexpr PointT operator-=(const PointT& rhs)
    {
		x -= rhs.x; y -= rhs.y;
		return *this;
    }

    constexpr PointT operator-(const PointT& rhs) const
    {
        PointT ret = *this;
        return ret -= rhs;
    }

	constexpr PointT operator*=( T mult )
	{
        x *= mult; y *= mult;
        return *this;
	}

    constexpr PointT operator*( T mult ) const
    {
        PointT ret = *this;
        return ret *= mult;
    }

    constexpr PointT operator/=( T div )
    {
        x /= div; y /= div;
        return *this;
    }

    constexpr PointT operator/( T div ) const
    {
        PointT ret = *this;
        return ret /= div;
    }

	//constexpr PointT( T _x, T _y ) : x(_x), y(_y) {}
	/// returns Euclidian distance to another point
	T Distance(PointT rhs) const
	{
		return sqrt((rhs.x - x) * (rhs.x - x) + (rhs.y - y) * (rhs.y - y));
	}
	/// returns squared Euclidian distance to another point
	T SquaredDistance(PointT rhs) const
	{
		return (rhs.x - x) * (rhs.x - x) + (rhs.y - y) * (rhs.y - y);
	}

	T Length() const
    {
        return sqrt(x * x + y * y);
    }

    T SquaredLength() const
    {
        return x * x + y * y;
    }

	bool operator==(const PointT& rhs) const 
	{
		return x == rhs.x && y == rhs.y;
	}

	bool operator!=(const PointT& rhs) const
	{
		return !(*this == rhs);
	}
	/// prints coords to a stream
	template <typename U>
	friend std::ostream& operator <<(std::ostream& out, const PointT<U>& point);

	template <typename U>
	friend PointT<U> operator*( U lhs, const PointT<U>& rhs );
};

template <typename T>
std::ostream& operator<<(std::ostream& out, const PointT<T>& point)
{
	return out << point.x << " " << point.y;
}

template <typename T>
PointT<T> operator*( T lhs, const PointT<T>& rhs )
{
    return PointT<T>(lhs * rhs.x, lhs * rhs.y);
}

/// neede for using points in hash tables
template <typename T>
struct PointTHasher
{
	std::size_t operator()(const PointT<T>& p) const
	{
		return std::hash<T>()(p.x) ^ (std::hash<T>()(p.y) << 1) >> 1;
	}
};

/// point with integer coords
using Point = PointT<int32_t>;
/// point with fractional coords
using PointF = PointT<float>;
/// point with fractional coords
using PointD = PointT<double>;
/// alias for Point hasher
using PointHasher = PointTHasher<int32_t>;
/// alias for PointD hasher
using PointDHasher = PointTHasher<double>;
/// alias for PointF hasher
using PointFHasher = PointTHasher<float>;


ACMB_NAMESPACE_END
