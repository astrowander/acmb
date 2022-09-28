#include "IPipelineElement.h"
#include "camerasettings.h"

ACMB_NAMESPACE_BEGIN

void IPipelineElement::CalcParams( std::shared_ptr<ImageParams> pParams )
{
    _width = pParams->GetWidth();
    _height = pParams->GetHeight();
    _pixelFormat = pParams->GetPixelFormat();
}

std::shared_ptr<CameraSettings> IPipelineElement::GetCameraSettings()
{
    return _pCameraSettings;
}

void IPipelineElement::SetCameraSettings( std::shared_ptr<CameraSettings> pCameraSettings )
{
    _pCameraSettings = pCameraSettings;
}

ACMB_NAMESPACE_END