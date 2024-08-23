#include "CenterObjectWindow.h"
#include "MainWindow.h"
#include "Serializer.h"
#include "ImGuiHelpers.h"
#include "./../Transforms/CenterObjectTransform.h"

ACMB_GUI_NAMESPACE_BEGIN

CenterObjectWindow::CenterObjectWindow( const Point& gridPos )
    : PipelineElementWindow( "Center Object", gridPos, PEFlags_StrictlyOneInput | PEFlags_StrictlyOneOutput )
{
}

void CenterObjectWindow::DrawPipelineElementControls()
{
    UI::DragFloat( "Threshold", &_threshold, 0.1f, 0.0f, 500.0f, "Threshold for object detecting" );
    UI::DragInt( "Crop Width", &_dstSize.width, 1.0f, 1, 65535, "Width of the cropped image" );
    UI::DragInt( "Crop Height", &_dstSize.height, 1.0f, 1, 65535, "Height of the cropped image" );
}

IBitmapPtr CenterObjectWindow::ProcessBitmapFromPrimaryInput( IBitmapPtr pSource, size_t )
{
    return CenterObjectTransform::CenterObject( pSource, {.dstSize = _dstSize, .threshold = _threshold } );
}

void CenterObjectWindow::Serialize( std::ostream& out ) const
{
    PipelineElementWindow::Serialize( out );
    gui::Serialize( _dstSize, out );
    gui::Serialize( _threshold, out );
}

bool CenterObjectWindow::Deserialize( std::istream& in )
{
    if ( !PipelineElementWindow::Deserialize( in ) ) return false;
    _dstSize = gui::Deserialize<Size>( in, _remainingBytes );
    _threshold = gui::Deserialize<float>( in, _remainingBytes );
    return true;
}

int CenterObjectWindow::GetSerializedStringSize() const
{
    return PipelineElementWindow::GetSerializedStringSize() + gui::GetSerializedStringSize( _dstSize ) + gui::GetSerializedStringSize( _threshold );
}

Expected<void, std::string> CenterObjectWindow::GeneratePreviewBitmap()
{
    _pPreviewBitmap = CenterObjectTransform::CenterObject( GetPrimaryInput()->GetPreviewBitmap()->Clone(), { .dstSize = _dstSize, .threshold = _threshold } );
    return {};
}

REGISTER_TOOLS_ITEM( CenterObjectWindow );

ACMB_GUI_NAMESPACE_END