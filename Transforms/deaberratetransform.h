#pragma once
#include "basetransform.h"
#include <functional>
#include "./../Tools/mathtools.h"

ACMB_NAMESPACE_BEGIN

struct CameraSettings;

class DeaberrateTransform : public BaseTransform
{
protected:
	std::shared_ptr<CameraSettings> _pCameraSettings;
	DeaberrateTransform( IBitmapPtr pSrcBitmap, std::shared_ptr<CameraSettings> pCameraSettings );
public:

	static std::shared_ptr<DeaberrateTransform> Create( IBitmapPtr pSrcBitmap, std::shared_ptr<CameraSettings> pCameraSettings );
};

ACMB_NAMESPACE_END
