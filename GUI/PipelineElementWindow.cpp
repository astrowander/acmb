#include "PipelineElementWindow.h"

ACMB_GUI_NAMESPACE_BEGIN

std::expected<IBitmapPtr, std::string> PipelineElementWindow::RunTaskAndReportProgress( size_t i )
{
    std::expected<IBitmapPtr, std::string> res;
    try
    {
        res =  RunTask( i );
    }
    catch ( std::exception& e )
    {
        res  = std::unexpected( e.what() );
    }

    _completedTaskCount = i + 1;
    return res;
}

void PipelineElementWindow::DrawDialog()
{
    DrawPipelineElementControls();
    ImGui::ProgressBar( _taskCount > 0 ? float( _completedTaskCount ) / float( _taskCount ) : 0.0f, { _itemWidth, 0 } );
    ImGui::Dummy( { -1, 0 } );
}

ACMB_GUI_NAMESPACE_END
