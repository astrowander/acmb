#pragma once
#include "./../Core/macros.h"
#include <cstdint>
#include <iostream>
#include <math.h>

ACMB_NAMESPACE_BEGIN

template<typename T>
struct PointT
{
	T x = 0;
	T y = 0;

	double Distance(PointT rhs) const
	{
		return sqrt((rhs.x - x) * (rhs.x - x) + (rhs.y - y) * (rhs.y - y));
	}

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

	template <typename U>
	friend std::ostream& operator <<(std::ostream& out, const PointT<U>& point);
};

template <typename T>
std::ostream& operator<<(std::ostream& out, const PointT<T>& point)
{
	return out << point.x << " " << point.y;
}

template <typename T>
struct PointTHasher
{
	std::size_t operator()(const PointT<T>& p) const
	{
		return std::hash<T>()(p.x) ^ (std::hash<T>()(p.y) << 1) >> 1;
	}
};

using Point = PointT<int32_t>;
using PointF = PointT<double>;
using PointHasher = PointTHasher<int32_t>;
using PointFHasher = PointTHasher<double>;

ACMB_NAMESPACE_END
