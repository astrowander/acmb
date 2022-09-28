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

    std::shared_ptr<ImageParams> GetFinalParams() const;
    std::shared_ptr<CameraSettings> GetCameraSettings() const;

    std::string GetFileName() const;
};
ACMB_NAMESPACE_END
