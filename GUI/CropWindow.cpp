#include "CropWindow.h"
#include "MainWindow.h"
#include "Serializer.h"
#include "./../Transforms/CropTransform.h"

ACMB_GUI_NAMESPACE_BEGIN

CropWindow::CropWindow( const Point& gridPos )
    : PipelineElementWindow( "Crop", gridPos, PEFlags_StrictlyOneInput | PEFlags_StrictlyOneOutput )
{
}

void CropWindow::DrawPipelineElementControls()
{
    ImGui::DragInt( "Left", &_dstRect.x, 1.0f, 1, 65535 );
    ImGui::DragInt( "Top", &_dstRect.y, 1.0f, 1, 65535 );
    ImGui::DragInt( "Width", &_dstRect.width, 1.0f, 1, 65535 );
    ImGui::DragInt( "Height", &_dstRect.height, 1.0f, 1, 65535 );
}

IBitmapPtr CropWindow::ProcessBitmapFromPrimaryInput( IBitmapPtr pSource, size_t )
{
    return CropTransform::Crop( pSource, _dstRect );
}

void CropWindow::Serialize( std::ostream& out )
{
    PipelineElementWindow::Serialize( out );
    gui::Serialize( _dstRect, out );
}

void CropWindow::Deserialize( std::istream& in )
{
    PipelineElementWindow::Deserialize( in );
    _dstRect = gui::Deserialize<Rect>( in, _remainingBytes );
}

int CropWindow::GetSerializedStringSize()
{
    return PipelineElementWindow::GetSerializedStringSize() + gui::GetSerializedStringSize( _dstRect );
}

REGISTER_TOOLS_ITEM( CropWindow );

ACMB_GUI_NAMESPACE_END
