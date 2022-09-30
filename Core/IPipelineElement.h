#pragma once
#include "macros.h"
#include "bitmap.h"
#include <vector>

ACMB_NAMESPACE_BEGIN

struct CameraSettings;

class IPipelineElement : public ImageParams
{
protected:
    std::shared_ptr<CameraSettings> _pCameraSettings;

public:
    virtual void CalcParams( std::shared_ptr<ImageParams> pParams );

    virtual IBitmapPtr ProcessBitmap( IBitmapPtr pSrcBitmap = nullptr ) = 0;
    virtual ~IPipelineElement() = default;

    std::shared_ptr<CameraSettings> GetCameraSettings() const;
    void SetCameraSettings( std::shared_ptr<CameraSettings> pCameraSettings );
};

class IPipelineFirstElement : public IPipelineElement
{
};

using IPipelineElementPtr = std::shared_ptr<IPipelineElement>;
using IPipelineFirstElementPtr = std::shared_ptr<IPipelineFirstElement>;

ACMB_NAMESPACE_END
