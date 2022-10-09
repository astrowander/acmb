#pragma once
#include "IPipelineElement.h"
ACMB_NAMESPACE_BEGIN

class BaseTransform;
/// <summary>
/// represent pipeline api, when bitmap passes through several transforms in a row
/// </summary>
class Pipeline
{
    std::vector<IPipelineElementPtr> _elements;

public:
    Pipeline() = default;
    /// creates pipeline with the given first element
    Pipeline( IPipelineFirstElementPtr pElement );
    /// adds given element to the pipeline
    void Add( IPipelineElementPtr pElement );    
    /// adds given transform to the pipeline, you can specify settings
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
    /// runs pipeline and returns result
    IBitmapPtr RunAndGetBitmap();
    /// calculates parameters of resulting image without running
    std::shared_ptr<ImageParams> GetFinalParams() const;
    /// returns camera settings
    std::shared_ptr<CameraSettings> GetCameraSettings() const;
    /// returns file name if the first element is a file decoder. Otherwise returns empty string
    std::string GetFileName() const;
    /// returns number of elements
    size_t GetSize() const;

    IPipelineElementPtr operator[]( size_t i )
    {
        return _elements[i];
    }
};

ACMB_NAMESPACE_END
