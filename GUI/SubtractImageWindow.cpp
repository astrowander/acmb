#include "SubtractImageWindow.h"
#include "MainWindow.h"
#include "Serializer.h"
#include "../Transforms/BitmapSubtractor.h"

ACMB_GUI_NAMESPACE_BEGIN

SubtractImageWindow::SubtractImageWindow( const Point& gridPos )
: PipelineElementWindow( "Subtract Dark Frame", gridPos, PEFlags::PEFlags_StrictlyTwoInputs | PEFlags::PEFlags_StrictlyOneOutput )
{
}

void SubtractImageWindow::DrawPipelineElementControls()
{
    ImGui::Checkbox( "Dark Frame is on Left", &_primaryInputIsOnTop );
}

IBitmapPtr SubtractImageWindow::ProcessBitmapFromPrimaryInput( IBitmapPtr pSource, size_t )
{
    return BitmapSubtractor::Subtract( pSource, _pSecondaryInputResult );
}

REGISTER_TOOLS_ITEM( SubtractImageWindow )

ACMB_GUI_NAMESPACE_END