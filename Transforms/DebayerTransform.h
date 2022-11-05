#pragma once
#include "basetransform.h"
#include "./../Tools/mathtools.h"

ACMB_NAMESPACE_BEGIN

/// <summary>
/// Creates debayered color image from undebayered one
/// </summary>
class DebayerTransform : public BaseTransform
{
public:
	using Settings = std::shared_ptr<CameraSettings>;
	
private:
	std::shared_ptr<CameraSettings> _pCameraSettings;

public:
	DebayerTransform( IBitmapPtr pSrcBitmap, std::shared_ptr<CameraSettings> pCameraSettings );
	/// Creates instance with source bitmap and camera settings
	static std::shared_ptr<DebayerTransform> Create( IBitmapPtr pSrcBitmap, std::shared_ptr<CameraSettings> pCameraSettings );
	/// Creates instance with source pixel format and camera settings. Source bitmap must be set later
	static std::shared_ptr<DebayerTransform> Create( PixelFormat pixelFormat, std::shared_ptr<CameraSettings> pCameraSettings );

	static IBitmapPtr Debayer( IBitmapPtr pSrcBitmap, std::shared_ptr<CameraSettings> pCameraSettings );

	virtual void Run() override;
	virtual void CalcParams( std::shared_ptr<ImageParams> pParams ) override;
	virtual void ValidateSettings() override;
};

ACMB_NAMESPACE_END
