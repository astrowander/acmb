#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "rect.h"
#include <array>

struct Triangle
{
	std::array<PointF, 3> vertices;

	RectF GetBoundingBox() const;

	bool IsPointInside(PointF p) const;

	double SquaredDistanceFromPoint(PointF p) const;
};
#endif // !TRIANGLE_H

