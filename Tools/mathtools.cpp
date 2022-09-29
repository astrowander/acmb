 #include "mathtools.h"

ACMB_NAMESPACE_BEGIN

float QuadraticInterpolation(float t, float t0, float t1, float t2)
{
    auto a = (t0 + t2) / 2 - t1;
    auto b = -(3 * t0 + t2) / 2 + 2 * t1;
    //auto c = t0;
    auto res = a * t * t + b * t + t0;
    return res;
}

float ArbitraryQuadraticInterpolation( float x, float x0, float y0, float x1, float y1, float x2, float y2 )
{
	const float x0sq = x0 * x0;
	const float x1sq = x1 * x1;
	const float x2sq = x2 * x2;
	float det = x0sq * x1 + x0 * x2sq + x1sq * x2 - x1 * x2sq - x0sq * x2 - x0 * x1sq;

	if ( fabs( det ) < std::numeric_limits<float>::epsilon() )
	{
		if ( x0 == x2 )
			return std::numeric_limits<float>::quiet_NaN();

		return y0 + ( x - x0 ) * ( y2 - y0 ) / ( x2 - x0 );
	}

	float detA = y0 * x1 + x0 * y2 + y1 * x2 - x1 * y2 - y0 * x2 - x0 * y1;
	float detB = x0sq * y1 + y0 * x2sq + x1sq * y2 - y1 * x2sq - x0sq * y2 - y0 * x1sq;
	float detC = x0sq * x1 * y2 + x0 * x2sq * y1 + x1sq * x2 * y0 - x1 * x2sq * y0 - x0sq * x2 * y1 - x0 * x1sq * y2;

	float res = ( detA * x * x + detB * x + detC ) / det;
	return res;
}

float NormalDist( float x, float xmax, float ymax, float sigma )
{
    return ymax * exp( -0.5f * ( xmax - x ) * ( xmax - x ) / sigma / sigma );
}

ACMB_NAMESPACE_END