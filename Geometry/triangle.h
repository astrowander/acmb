#pragma once

#include "rect.h"
#include <array>
#include <optional>

ACMB_NAMESPACE_BEGIN

/// <summary>
/// triangle with fractional coords
/// </summary>
struct Triangle
{
	std::array<PointD, 3> vertices;
	/// returns bounding rectagle
	RectD GetBoundingBox() const;
	/// checks if given point lies inside the triangle
	bool IsPointInside(const PointD& p) const;

    bool IsPointInsideUnchecked(const PointD& p) const;

	/// returns squared distance from the given point to the nearest edge
	double SquaredDistanceFromPoint(const PointD& p) const;
	/// returns barycenter point
	PointD GetCenter() const;

	std::optional<std::array<double, 3>> GetBarycentricCoords(const PointD& p) const;
};

ACMB_NAMESPACE_END
