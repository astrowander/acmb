#include "FlatFieldWindow.h"
#include "MainWindow.h"
#include "Serializer.h"
#include "ImGuiHelpers.h"

#include "./../Transforms/BitmapDivisor.h"

ACMB_GUI_NAMESPACE_BEGIN

FlatFieldWindow::FlatFieldWindow( const Point& gridPos )
    : PipelineElementWindow( "Apply Flat Field", gridPos, PEFlags::PEFlags_StrictlyTwoInputs | PEFlags::PEFlags_StrictlyOneOutput )
{
}

void FlatFieldWindow::DrawPipelineElementControls()
{
    UI::Checkbox( "Flat Frame is on Left", &_primaryInputIsOnTop, "By default the top image is considered as a flat field and applied to the left one. If checked, the left image is considered as a flat field" );
    UI::DragFloat( "Intensity", &_intensity, 0.1f, 0.0f, 500.0f, "The effect of the instrument can be weakened or enhanced. The default value is 100 percent", this );
}

IBitmapPtr FlatFieldWindow::ProcessBitmapFromPrimaryInput( IBitmapPtr pSource, size_t )
{
    return BitmapDivisor::Divide( pSource, { .pDivisor = _pSecondaryInputResult, .intensity = _intensity } );
}

void FlatFieldWindow::Serialize(std::ostream& out) const
{
    PipelineElementWindow::Serialize(out);
    gui::Serialize(_intensity, out);
}

void FlatFieldWindow::Deserialize(std::istream& in)
{
    PipelineElementWindow::Deserialize(in);
    _intensity = gui::Deserialize<float>(in, _remainingBytes);
}

int FlatFieldWindow::GetSerializedStringSize() const
{
    return PipelineElementWindow::GetSerializedStringSize() + gui::GetSerializedStringSize( _intensity );
}

Expected<void, std::string> FlatFieldWindow::GeneratePreviewBitmap()
{
    _pPreviewBitmap = BitmapDivisor::Divide( GetPrimaryInput()->GetPreviewBitmap()->Clone(), { .pDivisor = GetSecondaryInput()->GetPreviewBitmap(), .intensity = _intensity});
    return {};
}

REGISTER_TOOLS_ITEM( FlatFieldWindow );

ACMB_GUI_NAMESPACE_END