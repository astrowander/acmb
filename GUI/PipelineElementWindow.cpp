#include "PipelineElementWindow.h"

ACMB_GUI_NAMESPACE_BEGIN

std::expected<IBitmapPtr, std::string> PipelineElementWindow::RunTaskAndReportProgress( size_t i )
{
    _completedTaskCount = i + 1;

    try
    {
        return RunTask( i );
    }
    catch ( std::exception& e )
    {
        return std::unexpected( e.what() );
    }
}

void PipelineElementWindow::DrawDialog()
{
    DrawPipelineElementControls();
    ImGui::ProgressBar( _taskCount > 0 ? float( _completedTaskCount ) / float( _taskCount ) : 0.0f );
}

ACMB_GUI_NAMESPACE_END
