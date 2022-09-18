#define _USE_MATH_DEFINES
#include "startrektransform.h"

ACMB_NAMESPACE_BEGIN

StarTrekTransform::StarTrekTransform(const agg::trans_affine& affineMatrix, double delta0, double timeSpan)
: _affineMatrix(affineMatrix)
, _decl0(delta0)
, _timeSpan(timeSpan)
, _sinDecl0(sin(_decl0))
, _cosDecl0(cos(_decl0))
{

}

double StarTrekTransform::CosC(Vector2 sp)
{
	return _sinDecl0 * sin(sp[1]) + _cosDecl0 * cos(sp[1]) * cos(sp[0]);
}

double StarTrekTransform::XProjection(Vector2 sp)
{
	return cos(sp[1]) * sin(sp[0]) / CosC(sp);
}

double StarTrekTransform::YProjection(Vector2 sp)
{
	return (_cosDecl0 * sin(sp[1]) - _sinDecl0 * cos(sp[1]) * cos(sp[0])) / CosC(sp);
}

PointF StarTrekTransform::GetProjection(SphericalPoint sp)
{
	return { XProjection({sp.rha, sp.decl}), YProjection({sp.rha, sp.decl}) };
}

SphericalPoint StarTrekTransform::GetInverseProjection(PointF p/*, SphericalPoint firstApproach*/)
{
	auto res = Newton2D::Solve
	(
		FuncVector2
		{
			[this, p](auto x) {return this->XProjection(x) - p.x; },
			[this, p](auto x) {return this->YProjection(x) - p.y; }
		},
		{
			0,
			_decl0
		}
	);

	return { res[0], res[1] };
}

PointF StarTrekTransform::Transform(PointF p)
{
	_affineMatrix.transform(&p.x, &p.y);
	auto sp = GetInverseProjection(p);
	sp.rha += (_timeSpan / siderealDay) * 2 * M_PI;
	p = GetProjection(sp);
	_affineMatrix.inverse_transform(&p.x, &p.y);
	return p;
}

void StarTrekTransform::transform(double* x, double* y)
{
	PointF p { *x, *y };
	p = Transform(p);
	*x = p.x;
	*y = p.y;
}

ACMB_NAMESPACE_END