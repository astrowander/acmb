#ifndef STARTREKTRANFORM_H
#define STARTREKTRANFORM_H
#include "./../AGG/agg_trans_affine.h"


/*struct CameraSettings
{
	time_point<system_clock> shotTime;
	SizeF sensorSizeMm;
	double focalLength;	
};*/
struct SphericalPoint
{
	double rha;
	double decl;
};

class StarTrekTransform
{
	agg::trans_affine _affineMatrix;
	double _delta0 = 0.0;
	double _timeSpan = 0.0;

	StarTrekTransform(const agg::trans_affine& affineMatrix, double delta0, double timeSpan)
	: _affineMatrix(affineMatrix)
	, _delta0(delta0)
	, _timeSpan(timeSpan)
	{

	}

	
};

#endif

