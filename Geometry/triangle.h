#pragma once

#include "rect.h"
#include <array>

ACMB_NAMESPACE_BEGIN

struct Triangle
{
	std::array<PointF, 3> vertices;

	RectF GetBoundingBox() const;

	bool IsPointInside(PointF p) const;

	double SquaredDistanceFromPoint(PointF p) const;

	PointF GetCenter() const;
};

ACMB_NAMESPACE_END
