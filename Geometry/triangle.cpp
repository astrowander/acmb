#include "triangle.h"
ACMB_NAMESPACE_BEGIN

double Sign(const PointD& p1, const PointD& p2, const PointD& p3)
{
	return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y);
}

RectD Triangle::GetBoundingBox() const
{
	double x = std::min(vertices[0].x, std::min(vertices[1].x, vertices[2].x));
	double y = std::min(vertices[0].y, std::min(vertices[1].y, vertices[2].y));
	return RectD{ x, y, std::max(vertices[0].x, std::max(vertices[1].x, vertices[2].x)) - x, std::max(vertices[0].y, std::max(vertices[1].y, vertices[2].y)) - y };
}

bool Triangle::IsPointInside(const PointD& p) const
{
	if (!GetBoundingBox().IsPointInside(p))
		return false;

    return IsPointInsideUnchecked(p);
}

bool Triangle::IsPointInsideUnchecked(const PointD& p) const
{
	constexpr double epsilon = 1e-6;

	auto d1 = Sign(p, vertices[0], vertices[1]);
	auto d2 = Sign(p, vertices[1], vertices[2]);
	auto d3 = Sign(p, vertices[2], vertices[0]);

	bool hasNeg = (d1 < -epsilon) || (d2 < -epsilon) || (d3 < -epsilon);
	bool hasPos = (d1 > epsilon) || (d2 > epsilon) || (d3 > epsilon);

	return !(hasNeg && hasPos);
}

double Triangle::SquaredDistanceFromPoint(const PointD& p) const
{
	return std::min
	(
		p.SquaredDistance({ (vertices[0].x + vertices[1].x) / 2.0, (vertices[0].y + vertices[1].y) / 2.0 }),
		std::min
		(
			p.SquaredDistance({ (vertices[1].x + vertices[2].x) / 2.0, (vertices[1].y + vertices[2].y) / 2.0 }),
			p.SquaredDistance({ (vertices[2].x + vertices[0].x) / 2.0, (vertices[2].y + vertices[0].y) / 2.0 })
		)
	);
}

PointD Triangle::GetCenter() const
{
	return  { (vertices[0].x + vertices[1].x + vertices[2].x) / 3.0, (vertices[0].y + vertices[1].y + vertices[2].y) / 3.0 };
}

std::optional<std::array<double, 3>> Triangle::GetBarycentricCoords(const PointD& p) const
{
    const PointD& a = vertices[0];
    const PointD& b = vertices[1];
    const PointD& c = vertices[2];
    
	const double denominator = (b.y - c.y) * (a.x - c.x) + (c.x - b.x) * (a.y - c.y);
	if (denominator == 0.0)
		return std::nullopt; // Degenerate triangle

	const double alpha = ((b.y - c.y) * (p.x - c.x) + (c.x - b.x) * (p.y - c.y)) / denominator;
	const double beta = ((c.y - a.y) * (p.x - c.x) + (a.x - c.x) * (p.y - c.y)) / denominator;
	const double gamma = 1.0 - alpha - beta;

    return std::array<double, 3>{alpha, beta, gamma};
}

ACMB_NAMESPACE_END
