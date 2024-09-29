#pragma once

#include "./../AGG/agg_trans_affine.h"
#include "./point.h"
#include "./../Tools/Newton2D.h"

ACMB_NAMESPACE_BEGIN
/// point on a celestial sphere
struct SphericalPoint
{
	double rha;
	double decl;
};
/// <summary>
/// Star trails have the complicated form and motion of the stars cannot be represented with simple affine matrix. 
/// Here we have two shots of starry sky taken with known time interval, from still camera
/// We know pixel coords of a given star on the first image
/// This class calculates coords of this star on the second image
/// </summary>
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

	PointD GetProjection(SphericalPoint sp);
	SphericalPoint GetInverseProjection(PointD p/*, SphericalPoint firstApproach*/);

public:
	/// <summary>
	/// creates transform object with given params
	/// </summary>
	/// <param name="affineMatrix">basic affine matrix, defines scaling and rotation</param>
	/// <param name="delta0">declination in the center of image</param>
	/// <param name="timeSpan">time between thwo shots</param>
	StarTrekTransform(const agg::trans_affine& affineMatrix, double delta0, double timeSpan);
	/// receives coords of the star in first image, returns coords of the star in second image
	PointD Transform(PointD p);
	//for compatibility with agg::trans_affine
	void transform(double* x, double* y);
};

ACMB_NAMESPACE_END

