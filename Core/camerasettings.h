#pragma once
#include "./../Geometry/rect.h"

class CameraSettings
{
protected:
	SizeF _sensorSizeMm = {};
	double _focalLength = 0.0;
	double _radiansPerPixel = 0.0;
	int64_t _timestamp;

public:

	CameraSettings() = default;

	SizeF GetSensorSizeMm()
	{
		return _sensorSizeMm;
	}

	double GetFocalLength()
	{
		return _focalLength;
	}

	int64_t GetTimestamp()
	{
		return _timestamp;
	}

	double GetRadiansPerPixel()
	{
		return _radiansPerPixel;
	}
};
