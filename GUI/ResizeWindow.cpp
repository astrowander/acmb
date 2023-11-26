#include "ResizeWindow.h"
#include "MainWindow.h"
#include "Serializer.h"
#include "ImGuiHelpers.h"
#include "./../Transforms/ResizeTransform.h"
ACMB_GUI_NAMESPACE_BEGIN

ResizeWindow::ResizeWindow( const Point& gridPos )
    : PipelineElementWindow( "Resize", gridPos, PEFlags_StrictlyOneInput | PEFlags_StrictlyOneOutput )
{
}

void ResizeWindow::DrawPipelineElementControls()
{
    ImGui::Text( "Destination Size" );
    UI::DragInt( "Width", &_dstSize.width, 1.0f, 1, 65535, "Width of the resized image" );
    UI::DragInt( "Height", &_dstSize.height, 1.0f, 1, 65535, "Height of the resized image" );
}

IBitmapPtr ResizeWindow::ProcessBitmapFromPrimaryInput( IBitmapPtr pSource, size_t )
{
    return ResizeTransform::Resize( pSource, _dstSize );
}

void ResizeWindow::Serialize( std::ostream& out ) const
{
    PipelineElementWindow::Serialize( out );
    gui::Serialize( _dstSize, out );
}

void ResizeWindow::Deserialize( std::istream& in )
{
    PipelineElementWindow::Deserialize( in );
    _dstSize = gui::Deserialize<Size>( in, _remainingBytes );
}

int ResizeWindow::GetSerializedStringSize() const
{
    return PipelineElementWindow::GetSerializedStringSize() + gui::GetSerializedStringSize( _dstSize );
}

REGISTER_TOOLS_ITEM( ResizeWindow )

ACMB_GUI_NAMESPACE_END