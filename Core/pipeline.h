#pragma once
#include "IPipelineElement.h"
ACMB_NAMESPACE_BEGIN

class BaseTransform;
class Pipeline
{
    std::vector<IPipelineElementPtr> _elements;

public:
    Pipeline() = default;
    Pipeline( IPipelineFirstElementPtr pElement );

    void Add( IPipelineElementPtr pElement );    

    template<std::derived_from<BaseTransform> ElementType>
    void AddTransform( typename ElementType::Settings settings = {})
    {
        if ( _elements.empty() )
            throw std::runtime_error( "unable to add a transform to the empty pipeline" );

        auto pElement = ElementType::Create( GetFinalParams()->GetPixelFormat(), settings );
        pElement->CalcParams( _elements.back() );
        pElement->SetCameraSettings( _elements.back()->GetCameraSettings() );
        _elements.push_back( pElement );        
    }

    IBitmapPtr RunAndGetBitmap();

    std::shared_ptr<ImageParams> GetFinalParams() const;
    std::shared_ptr<CameraSettings> GetCameraSettings() const;

    std::string GetFileName() const;
    size_t GetSize() const;
};

ACMB_NAMESPACE_END
