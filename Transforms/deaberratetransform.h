#pragma once
#include "basetransform.h"
#include <functional>
#include "./../Tools/mathtools.h"

ACMB_NAMESPACE_BEGIN

struct CameraSettings;
/// <summary>
/// Fixes optical aberration in the image if lens parameters are known
/// </summary>
class DeaberrateTransform : public BaseTransform
{
public:
	using Settings = std::shared_ptr<CameraSettings>;
protected:
	std::shared_ptr<CameraSettings> _pCameraSettings;
	DeaberrateTransform( IBitmapPtr pSrcBitmap, std::shared_ptr<CameraSettings> pCameraSettings );
public:
	/// Creates instance with source bitmap and camera settings
	static std::shared_ptr<DeaberrateTransform> Create( IBitmapPtr pSrcBitmap, std::shared_ptr<CameraSettings> pCameraSettings );
	/// Creates instance with source pixel format and camera settings. Source bitmap must be set later
	static std::shared_ptr<DeaberrateTransform> Create( PixelFormat pixelFormat, std::shared_ptr<CameraSettings> pCameraSettings );
};

ACMB_NAMESPACE_END
