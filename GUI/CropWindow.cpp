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
    ImGui::Text( "Crop area:" );
    UI::DragInt( "Left", &_dstRect.x, 1.0f, 1, 65535, "Left boundary of the crop area" );
    UI::DragInt( "Top", &_dstRect.y, 1.0f, 1, 65535,"Top boundary of the crop area" );
    UI::DragInt( "Width", &_dstRect.width, 1.0f, 1, 65535,  "Width of the crop area" );
    UI::DragInt( "Height", &_dstRect.height, 1.0f, 1, 65535, "Height of the crop area" );
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
    const auto inputSizeExp = GetBitmapSize();
    if ( !inputSizeExp )
        return unexpected( inputSizeExp.error() );

    const Size inputSize = inputSizeExp.value();
    const float xFactor = float(inputPreviewSize.width) / float( inputSize.width );
    const float yFactor = float(inputPreviewSize.height) / float( inputSize.height );

    const Rect cropArea{ .x = int( _dstRect.x * xFactor ), .y = int( _dstRect.y * yFactor ), .width = int( _dstRect.width * xFactor ), .height = int( _dstRect.height * yFactor ) };
    _pPreviewBitmap = CropTransform::Crop( GetPrimaryInput()->GetPreviewBitmap()->Clone(), cropArea );
    return {};
}

REGISTER_TOOLS_ITEM( CropWindow );

ACMB_GUI_NAMESPACE_END
