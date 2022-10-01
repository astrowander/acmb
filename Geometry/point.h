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
	/// returns Euclidian distance to another point
	double Distance(PointT rhs) const
	{
		return sqrt((rhs.x - x) * (rhs.x - x) + (rhs.y - y) * (rhs.y - y));
	}
	/// returns squared Euclidian distance to another point
	double SquaredDistance(PointT rhs) const
	{
		return (rhs.x - x) * (rhs.x - x) + (rhs.y - y) * (rhs.y - y);
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
};

template <typename T>
std::ostream& operator<<(std::ostream& out, const PointT<T>& point)
{
	return out << point.x << " " << point.y;
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
using PointF = PointT<double>;
/// alias for Point hasher
using PointHasher = PointTHasher<int32_t>;
/// alias for PointF hasher
using PointFHasher = PointTHasher<double>;

ACMB_NAMESPACE_END
