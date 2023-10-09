#include "SubtractImageWindow.h"
#include "MainWindow.h"
#include "../Transforms/BitmapSubtractor.h"

ACMB_GUI_NAMESPACE_BEGIN

SubtractImageWindow::SubtractImageWindow( const Point& gridPos )
: PipelineElementWindow( "Subtract Dark Frame", gridPos, PEFlags::PEFlags_StrictlyTwoInputs | PEFlags::PEFlags_StrictlyOneOutput )
{
}

void SubtractImageWindow::DrawPipelineElementControls()
{
    ImGui::Text( "Dark Frame Is on:" );
    ImGui::RadioButton( "Top", &_darkFrameIsOnTop, 1 );
    ImGui::RadioButton( "Left", &_darkFrameIsOnTop, 0 );
}

std::expected<IBitmapPtr, std::string> SubtractImageWindow::RunTask( size_t i )
{
    if ( _completedTaskCount == 0 )
    {
        const auto pDarkFrameWindow = ( _darkFrameIsOnTop ) ? GetTopInput() : GetLeftInput();
        if ( !pDarkFrameWindow )
            return std::unexpected( "No dark frame" );

        const auto darkFrameRes = pDarkFrameWindow->RunTaskAndReportProgress( 0 );
        if ( !darkFrameRes )
            return darkFrameRes;

        _pBitmapToSubtract = *darkFrameRes;
    }

    const auto pInput = ( _darkFrameIsOnTop ) ? GetLeftInput() : GetTopInput();
    if ( !pInput )
        return std::unexpected( "No input " );

    if ( _taskCount == 0 )
    {
        _taskCount = pInput->GetTaskCount();
    }

    const auto taskRes = pInput->RunTaskAndReportProgress( i );
    if ( !taskRes.has_value() )
        return std::unexpected( taskRes.error() );

    return BitmapSubtractor::Subtract( *taskRes, _pBitmapToSubtract );
}

REGISTER_TOOLS_ITEM( SubtractImageWindow )

ACMB_GUI_NAMESPACE_END