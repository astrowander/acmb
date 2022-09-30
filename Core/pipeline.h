#pragma once
#include "IPipelineElement.h"
ACMB_NAMESPACE_BEGIN

class BaseTransform;
class Pipeline
{
    std::vector<IPipelineElementPtr> _elements;

public:
    Pipeline() = default;
    Pipeline( IPipelineElementPtr pElement );

    void Add( IPipelineElementPtr pElement );    

    template<std::derived_from<BaseTransform> ElementType>
    void AddTransform( typename ElementType::Settings settings = {})
    {
        _elements.push_back( ElementType::Create( GetFinalParams()->GetPixelFormat(), settings ) );
    }

    IBitmapPtr RunAndGetBitmap();

    std::shared_ptr<ImageParams> GetFinalParams() const;
    std::shared_ptr<CameraSettings> GetCameraSettings() const;

    std::string GetFileName() const;
    size_t GetSize() const;
};

ACMB_NAMESPACE_END
