#include "triangle.h"

float Sign(PointF p1, PointF p2, PointF p3)
{
	return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y);
}

RectF Triangle::GetBoundingBox() const
{
	double x = std::min(vertices[0].x, std::min(vertices[1].x, vertices[2].x));
	double y = std::min(vertices[0].y, std::min(vertices[1].y, vertices[2].y));
	return RectF{ x, y, std::max(vertices[0].x, std::max(vertices[1].x, vertices[2].x)) - x, std::max(vertices[0].y, std::max(vertices[1].y, vertices[2].y)) - y };
}

bool Triangle::IsPointInside(PointF p) const
{
	if (!GetBoundingBox().IsPointInside(p))
		return false;

	auto d1 = Sign(p, vertices[0], vertices[1]);
	auto d2 = Sign(p, vertices[1], vertices[2]);
	auto d3 = Sign(p, vertices[2], vertices[0]);

	bool hasNeg = (d1 < 0) || (d2 < 0) || (d3 < 0);
	bool hasPos = (d1 > 0) || (d2 > 0) || (d3 > 0);

	return !(hasNeg && hasPos);
}

double Triangle::SquaredDistanceFromPoint(PointF p) const
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

PointF Triangle::GetCenter() const
{
	return  { (vertices[0].x + vertices[1].x + vertices[2].x) / 3.0, (vertices[0].y + vertices[1].y + vertices[2].y) / 3.0 };
}
