#ifndef STARTREKTRANSFORM_H
#define STARTREKTRANSFORM_H

#include "./../AGG/agg_trans_affine.h"
#include "./point.h"
#include "./../Tools/Newton2D.h"

struct SphericalPoint
{
	double rha;
	double decl;
};

class StarTrekTransform
{
	agg::trans_affine _affineMatrix;
	double _decl0;
	double _timeSpan;
	double _sinDecl0;
	double _cosDecl0;

	const double siderealDay = 86164.0;

	double CosC(Vector2 sp);
	double XProjection(Vector2 sp);
	double YProjection(Vector2 sp);

	PointF GetProjection(SphericalPoint sp);
	SphericalPoint GetInverseProjection(PointF p/*, SphericalPoint firstApproach*/);

public:

	StarTrekTransform(const agg::trans_affine& affineMatrix, double delta0, double timeSpan);
	PointF Transform(PointF p);
	//for compatibility with agg::trans_affine
	void transform(double* x, double* y);
};

#endif

