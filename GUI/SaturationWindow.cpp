#include "SaturationWindow.h"
#include "Serializer.h"
#include "MainWindow.h"
#include "ImGuiHelpers.h"

#include "./../Transforms/SaturationTransform.h"

ACMB_GUI_NAMESPACE_BEGIN

SaturationWindow::SaturationWindow( const Point& gridPos )
: PipelineElementWindow( "Saturation", gridPos, PEFlags_StrictlyOneInput | PEFlags_StrictlyOneOutput )
{
}

void SaturationWindow::DrawPipelineElementControls()
{
    UI::DragFloat( "Saturation", &_saturation, 0.01f, 0.0f, 4.0f, "Saturation factor", this );
}

Expected<void, std::string> SaturationWindow::GeneratePreviewBitmap()
{
    auto pInputBitmap = GetPrimaryInput()->GetPreviewBitmap()->Clone();
    _pPreviewBitmap = SaturationTransform::Saturate( pInputBitmap, _saturation );
    return {};
}

IBitmapPtr SaturationWindow::ProcessBitmapFromPrimaryInput( IBitmapPtr pSource, size_t taskNumber )
{
    return SaturationTransform::Saturate( pSource, _saturation );
}

void SaturationWindow::Serialize( std::ostream& out ) const
{
    PipelineElementWindow::Serialize( out );
    gui::Serialize( _saturation, out );
}

void SaturationWindow::Deserialize( std::istream& in )
{
    PipelineElementWindow::Deserialize( in );
    _saturation = gui::Deserialize<float>( in, _remainingBytes );
}

int SaturationWindow::GetSerializedStringSize() const
{
    return PipelineElementWindow::GetSerializedStringSize() 
    + gui::GetSerializedStringSize( _saturation );
}

REGISTER_TOOLS_ITEM( SaturationWindow );

ACMB_GUI_NAMESPACE_END
