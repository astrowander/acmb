#include "ResizeWindow.h"
#include "MainWindow.h"
#include "./../Transforms/ResizeTransform.h"
ACMB_GUI_NAMESPACE_BEGIN

ResizeWindow::ResizeWindow( const Point& gridPos )
: PipelineElementWindow( "Resize", gridPos, PEFlags_StrictlyOneInput | PEFlags_StrictlyOneOutput )
{
}

void ResizeWindow::DrawPipelineElementControls()
{
    ImGui::Text( "Destination Size" );
    ImGui::DragInt( "Width", &_dstSize.width, 1.0f, 1, 65535 );
    ImGui::DragInt( "Height", &_dstSize.height, 1.0f, 1, 65535 );
}

std::expected<IBitmapPtr, std::string> ResizeWindow::RunTask( size_t i )
{
    try
    {
        auto pPrimaryInput = GetLeftInput();
        if ( !pPrimaryInput )
            pPrimaryInput = GetTopInput();

        if ( !pPrimaryInput )
            return std::unexpected( "No input element" );

        if ( _taskCount == 0 )
        {
            _taskCount = pPrimaryInput->GetTaskCount();
        }

        const auto taskRes = pPrimaryInput->RunTaskAndReportProgress( i );
        if ( !taskRes.has_value() )
            return std::unexpected( taskRes.error() );

        return ResizeTransform::Resize( *taskRes, _dstSize );
    }
    catch ( std::exception& e )
    {
        return std::unexpected( e.what() );
    }
}

REGISTER_TOOLS_ITEM( ResizeWindow )

ACMB_GUI_NAMESPACE_END