#pragma once
#include "IPipelineElement.h"

ACMB_NAMESPACE_BEGIN

class Pipeline
{
    std::vector<IPipelineElementPtr> _elements;

public:
    Pipeline() = default;
    Pipeline( IPipelineElementPtr pElement );

    void Add( IPipelineElementPtr pElement );

    IBitmapPtr RunAndGetBitmap();

    std::shared_ptr<ImageParams> GetFinalParams();
    std::shared_ptr<CameraSettings> GetCameraSettings();

    std::string GetFileName();
};
ACMB_NAMESPACE_END
