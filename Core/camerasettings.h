#pragma once
#include "./../Geometry/rect.h"
#include <array>

ACMB_NAMESPACE_BEGIN

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

	uint16_t blackLevel = 0;
	uint16_t maxChannel = 0xFFFF;	
	std::array<float, 4> channelPremultipiers = { 1.0f, 1.0f, 1.0f, 1.0f };
};

ACMB_NAMESPACE_END