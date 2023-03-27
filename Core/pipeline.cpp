#include "pipeline.h"
#include "./../Codecs/imagedecoder.h"

ACMB_NAMESPACE_BEGIN

Pipeline::Pipeline( IPipelineFirstElementPtr pElement )
{
    if ( !pElement )
        throw std::invalid_argument( " pElement is null" );

    _elements.push_back( pElement );
}

void Pipeline::Add( IPipelineElementPtr pElement )
{
    if ( !pElement )
        throw std::invalid_argument( " pElement is null" );

    auto pFirstElement = std::dynamic_pointer_cast< IPipelineFirstElement >( pElement );
    if ( _elements.empty() && !pFirstElement )
    {
        throw std::runtime_error( "unable to add this element to the start of the pipeline" );
    }
    
    if ( !_elements.empty() )
    {
        if ( pFirstElement )
            throw std::runtime_error( "this element can be added only to the start of the pipeline" );

        pElement->CalcParams( _elements.back() );
        pElement->SetCameraSettings( _elements.back()->GetCameraSettings() );
    }

    _elements.push_back( pElement );
}

IBitmapPtr Pipeline::RunAndGetBitmap()
{
    if ( _elements.empty() )
        return nullptr;

    auto pBitmap = _elements[0]->ProcessBitmap();
    for ( size_t i = 1; i < _elements.size(); ++i )
    {
        _elements[i]->SetCameraSettings( _elements[i - 1]->GetCameraSettings() );
        pBitmap = _elements[i]->ProcessBitmap( pBitmap );     
    }
    return pBitmap;
}

std::shared_ptr<ImageParams> Pipeline::GetFinalParams() const
{
    return _elements.empty() ? nullptr : _elements.back();
}

std::shared_ptr<CameraSettings> Pipeline::GetCameraSettings() const
{
    return _elements.empty() ? nullptr : _elements.back()->GetCameraSettings();
}

std::string Pipeline::GetFileName() const
{
    if ( _elements.empty() )
        return {};

    auto pDecoder = std::dynamic_pointer_cast<ImageDecoder>( _elements[0] );
    if ( !pDecoder )
        return {};

    return pDecoder->GetLastFileName();
}

size_t Pipeline::GetSize() const
{
    return _elements.size();
}

ACMB_NAMESPACE_END
