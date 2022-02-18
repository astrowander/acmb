#pragma once
#include "./../Geometry/rect.h"

struct CameraSettings
{
	std::string cameraMakerName;
	std::string cameraModelName;
	std::string lensMakerName;
	std::string lensModelName;

	SizeF sensorSizeMm = {};
	double cropFactor = 1.0;
	double focalLength = 0.0;
	double radiansPerPixel = 0.0;
	double aperture = 0.0;
	double distance = 1000.0;

	int64_t timestamp;
};
