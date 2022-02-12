#ifndef POINT_H
#define POINT_H
#include <cstdint>
#include <math.h>

template<typename T>
struct PointT
{
	T x = 0;
	T y = 0;

	double Distance(PointT rhs)
	{
		return sqrt((rhs.x - x) * (rhs.x - x) + (rhs.y - y) * (rhs.y - y));
	}

	double SquaredDistance(PointT rhs)
	{
		return (rhs.x - x) * (rhs.x - x) + (rhs.y - y) * (rhs.y - y);
	}
};

using Point = PointT<int32_t>;
using PointF = PointT<double>;

#endif 
