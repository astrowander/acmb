#include "deaberratetransform.h"
#include "./../Core/camerasettings.h"



DeaberrateTransform::DeaberrateTransform(IBitmapPtr pSrcBitmap, std::shared_ptr<CameraSettings> pCameraSettings)
: BaseTransform(pSrcBitmap)
, _pDatabase(new lfDatabase())
, _pCameraSettings(pCameraSettings)
{
	if (_pDatabase->Load())
		throw std::runtime_error("unable to load the lens database");
}

void DeaberrateTransform::Run()
{
	auto cameras = std::unique_ptr<const lfCamera*, std::function<void(void*)>>(_pDatabase->FindCameras(_pCameraSettings->cameraMakerName.c_str(), _pCameraSettings->cameraModelName.c_str()), lf_free);	
	if (!cameras)
		throw std::runtime_error("unable to find the camera");

	auto pCamera = cameras.get()[0];

	auto lenses = std::unique_ptr<const lfLens*, std::function<void(void*)>>(_pDatabase->FindLenses(pCamera, _pCameraSettings->lensMakerName.c_str(), _pCameraSettings->lensModelName.c_str()), lf_free);
	if (!lenses)
		throw std::runtime_error("unable to find the lens");

	auto pLens = lenses.get()[0];

	uint32_t width = _pSrcBitmap->GetWidth();
	uint32_t height = _pSrcBitmap->GetHeight();
	const PixelFormat pixelFormat = _pSrcBitmap->GetPixelFormat();

	_pModifier.reset(new lfModifier(pLens, _pCameraSettings->focalLength, _pCameraSettings->cropFactor, width, height, BytesPerChannel(pixelFormat) == 1 ? LF_PF_U8 : LF_PF_U16));
	


	//auto modFlags = _pModifier->Initialize(pLens, BytesPerChannel(pixelFormat) == 1 ? LF_PF_U8 : LF_PF_U16, _pCameraSettings->focalLength, _pCameraSettings->aperture, _pCameraSettings->distance, 1.0, pLens->Type, LF_MODIFY_DISTORTION | LF_MODIFY_VIGNETTING, false);
	
	_pDstBitmap = IBitmap::Create(width, height, pixelFormat);
	
	//correct vignetting if available
	if (_pModifier->EnableVignettingCorrection(_pCameraSettings->aperture, _pCameraSettings->distance) & LF_MODIFY_VIGNETTING)
	{
		for (uint32_t y = 0; y < _pSrcBitmap->GetHeight(); y++)
		{
			auto pScanline = _pSrcBitmap->GetPlanarScanline(y);
			if (!_pModifier->ApplyColorModification(pScanline, 0.0, y, width, height, LF_CR_4(RED, GREEN, BLUE, UNKNOWN), width * BytesPerPixel(pixelFormat)))
				throw std::runtime_error("unable to correct vignetting");
		}
	}

	_pModifier->EnableDistortionCorrection();
	_pModifier->EnableTCACorrection();

	if ((_pModifier->GetModFlags() & LF_MODIFY_DISTORTION) || (_pModifier->GetModFlags() & LF_MODIFY_TCA))
	{
		//correct TCA and distortion
		if (pixelFormat == PixelFormat::RGB24)
			CorrectDistortion<PixelFormat::RGB24>();
		else
			CorrectDistortion<PixelFormat::RGB48>();
	}	
}
