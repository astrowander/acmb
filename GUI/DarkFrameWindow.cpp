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
}

IBitmapPtr DarkFrameWindow::ProcessBitmapFromPrimaryInput( IBitmapPtr pSource, size_t )
{
    return BitmapSubtractor::Subtract( pSource, _pSecondaryInputResult );
}

REGISTER_TOOLS_ITEM( DarkFrameWindow )

ACMB_GUI_NAMESPACE_END