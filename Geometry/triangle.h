#pragma once

#include "rect.h"
#include <array>

ACMB_NAMESPACE_BEGIN

/// <summary>
/// triangle with fractional coords
/// </summary>
struct Triangle
{
	std::array<PointD, 3> vertices;
	/// returns bounding rectagle
	RectF GetBoundingBox() const;
	/// checks if given point lies inside the triangle
	bool IsPointInside(PointD p) const;
	/// returns squared distance from the given point to the nearest edge
	double SquaredDistanceFromPoint(PointD p) const;
	/// returns barycenter point
	PointD GetCenter() const;
};

ACMB_NAMESPACE_END
