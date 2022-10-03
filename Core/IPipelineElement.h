#pragma once
#include "macros.h"
#include "bitmap.h"
#include <vector>

ACMB_NAMESPACE_BEGIN

struct CameraSettings;
/// <summary>
/// Abstract class for pipeline element
/// </summary>
class IPipelineElement : public ImageParams
{
protected:
    std::shared_ptr<CameraSettings> _pCameraSettings;

public:
    /// override this if derived class changes size or pixel format of the image
    virtual void CalcParams( std::shared_ptr<ImageParams> pParams );
    /// abstract fuction where the given bitmap is processed
    virtual IBitmapPtr ProcessBitmap( IBitmapPtr pSrcBitmap = nullptr ) = 0;
    virtual ~IPipelineElement() = default;
    /// returns camera settings
    std::shared_ptr<CameraSettings> GetCameraSettings() const;
    /// sets camera settings
    void SetCameraSettings( std::shared_ptr<CameraSettings> pCameraSettings );
};

/// empty class, inherit from it if derived objects can be only in the start of pipeline
class IPipelineFirstElement : public IPipelineElement
{
};
/// alias for pointer to pipeline element
using IPipelineElementPtr = std::shared_ptr<IPipelineElement>;
/// alias for pointer to first pipeline element
using IPipelineFirstElementPtr = std::shared_ptr<IPipelineFirstElement>;

ACMB_NAMESPACE_END