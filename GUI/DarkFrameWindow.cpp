#include "DarkFrameWindow.h"
#include "MainWindow.h"
#include "Serializer.h"
#include "../Transforms/BitmapSubtractor.h"
#include "ImGuiHelpers.h"

ACMB_GUI_NAMESPACE_BEGIN

DarkFrameWindow::DarkFrameWindow( const Point& gridPos )
: PipelineElementWindow( "Subtract Dark Frame", gridPos, PEFlags::PEFlags_StrictlyTwoInputs | PEFlags::PEFlags_StrictlyOneOutput )
{
}

void DarkFrameWindow::DrawPipelineElementControls()
{
    UI::Checkbox( "Dark Frame is on Left", &_primaryInputIsOnTop, "By default the top image is subtracted from the left one. If checked, the left image is subtracted from the top one" );
    UI::DragFloat( "Intensity", &_intensity, 0.1f, 0.0f, 500.0f, "The effect of the instrument can be weakened or enhanced. The default value is 100 percent" );
}

IBitmapPtr DarkFrameWindow::ProcessBitmapFromPrimaryInput( IBitmapPtr pSource, size_t )
{
    return BitmapSubtractor::Subtract( pSource, { .pBitmapToSubtract = _pSecondaryInputResult, .intensity = _intensity } );
}

void DarkFrameWindow::Serialize( std::ostream& out ) const
{
    PipelineElementWindow::Serialize( out );
    gui::Serialize( _intensity, out );
}

int DarkFrameWindow::GetSerializedStringSize() const
{
    return PipelineElementWindow::GetSerializedStringSize() + gui::GetSerializedStringSize( _intensity );
}

void DarkFrameWindow::Deserialize( std::istream& in )
{
    PipelineElementWindow::Deserialize( in );
    _intensity = gui::Deserialize<float>( in, _remainingBytes );
}

REGISTER_TOOLS_ITEM( DarkFrameWindow )

ACMB_GUI_NAMESPACE_END