#pragma once

#include "rect.h"
#include <array>

ACMB_NAMESPACE_BEGIN

/// <summary>
/// triangle with fractional coords
/// </summary>
struct Triangle
{
	std::array<PointF, 3> vertices;
	/// returns bounding rectagle
	RectF GetBoundingBox() const;
	/// checks if given point lies inside the triangle
	bool IsPointInside(PointF p) const;
	/// returns squared distance from the given point to the nearest edge
	double SquaredDistanceFromPoint(PointF p) const;
	/// returns barycenter point
	PointF GetCenter() const;
};

ACMB_NAMESPACE_END
