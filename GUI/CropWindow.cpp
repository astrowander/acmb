#include "CropWindow.h"
#include "MainWindow.h"
#include "Serializer.h"
#include "ImGuiHelpers.h"
#include "./../Transforms/CropTransform.h"

ACMB_GUI_NAMESPACE_BEGIN

CropWindow::CropWindow( const Point& gridPos )
    : PipelineElementWindow( "Crop", gridPos, PEFlags_StrictlyOneInput | PEFlags_StrictlyOneOutput )
{
}

void CropWindow::DrawPipelineElementControls()
{
    Size inputBitmapSize = { 65535, 65535 };
    if ( auto pPrimaryInput = GetPrimaryInput() )    
        if ( auto inputBitmapSizeExp = pPrimaryInput->GetBitmapSize() )
            inputBitmapSize = inputBitmapSizeExp.value();    

    ImGui::Text( "Crop area:" );
    UI::DragInt( "Left", &_dstRect.x, 1.0f, 0, inputBitmapSize.width - 1, "Left boundary of the crop area", this );
    UI::DragInt( "Top", &_dstRect.y, 1.0f, 0, inputBitmapSize.height - 1,"Top boundary of the crop area", this );
    UI::DragInt( "Width", &_dstRect.width, 1.0f, 1, inputBitmapSize.width - _dstRect.x,  "Width of the crop area", this );
    UI::DragInt( "Height", &_dstRect.height, 1.0f, 1, inputBitmapSize.height - _dstRect.y, "Height of the crop area", this );
}

IBitmapPtr CropWindow::ProcessBitmapFromPrimaryInput( IBitmapPtr pSource, size_t )
{
    return CropTransform::Crop( pSource, _dstRect );
}

void CropWindow::Serialize( std::ostream& out ) const
{
    PipelineElementWindow::Serialize( out );
    gui::Serialize( _dstRect, out );
}

void CropWindow::Deserialize( std::istream& in )
{
    PipelineElementWindow::Deserialize( in );
    _dstRect = gui::Deserialize<Rect>( in, _remainingBytes );
}

int CropWindow::GetSerializedStringSize() const
{
    return PipelineElementWindow::GetSerializedStringSize() + gui::GetSerializedStringSize( _dstRect );
}

Expected<void, std::string> CropWindow::GeneratePreviewBitmap()
{
    const auto pInputBitmap = GetPrimaryInput()->GetPreviewBitmap();
    const Size inputPreviewSize{ int( pInputBitmap->GetWidth() ), int( pInputBitmap->GetHeight() ) };
    const auto inputSizeExp = GetPrimaryInput()->GetBitmapSize();
    if ( !inputSizeExp )
        return unexpected( inputSizeExp.error() );

    const Size inputSize = inputSizeExp.value();
    const float xFactor = float(inputPreviewSize.width) / float( inputSize.width );
    const float yFactor = float(inputPreviewSize.height) / float( inputSize.height );

    const Rect cropArea
    { 
        .x = std::clamp( int( _dstRect.x * xFactor ), 0, inputPreviewSize.width - 1 ),
        .y = std::clamp( int( _dstRect.y * yFactor ), 0, inputPreviewSize.height - 1 ),
        .width = std::clamp( int( _dstRect.width * xFactor ), 1, inputPreviewSize.width - cropArea.x ),
        .height = std::clamp( int( _dstRect.height * yFactor ), 1, inputPreviewSize.height - cropArea.y )
    };
    _pPreviewBitmap = CropTransform::Crop( GetPrimaryInput()->GetPreviewBitmap()->Clone(), cropArea );
    return {};
}

Expected<Size, std::string> CropWindow::GetBitmapSize()
{
    return Size{ _dstRect.width, _dstRect.height };
}

REGISTER_TOOLS_ITEM( CropWindow );

ACMB_GUI_NAMESPACE_END
