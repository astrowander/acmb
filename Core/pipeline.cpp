#include "pipeline.h"
#include "./../Codecs/imagedecoder.h"

ACMB_NAMESPACE_BEGIN

Pipeline::Pipeline( IPipelineElementPtr pElement )
{
    _elements.push_back( pElement );
}

void Pipeline::Add( IPipelineElementPtr pElement )
{
    if ( !_elements.empty() )
    {
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
