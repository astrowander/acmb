#include "PipelineElementWindow.h"

ACMB_GUI_NAMESPACE_BEGIN

PipelineElementWindow::PipelineElementWindow( const std::string& name, const Point& gridPos, int inOutFlags )
: Window( name + "##R" + std::to_string( gridPos.y ) + "C" + std::to_string( gridPos.x ), { cElementWidth, cElementHeight } )
, _inOutFlags( inOutFlags )
, _itemWidth( cElementWidth - ImGui::GetStyle().WindowPadding.x * cMenuScaling )
{
}

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

std::shared_ptr<PipelineElementWindow>  PipelineElementWindow::GetLeftInput()
{
    return _pLeftInput.lock();
}

std::shared_ptr<PipelineElementWindow>  PipelineElementWindow::GetTopInput()
{
    return _pTopInput.lock();
}

void PipelineElementWindow::SetLeftInput( std::shared_ptr<PipelineElementWindow> pPrimaryInput )
{
    _pLeftInput = pPrimaryInput;
}

void PipelineElementWindow::SetTopInput( std::shared_ptr<PipelineElementWindow> pSecondaryInput )
{
    _pTopInput = pSecondaryInput;
}

std::shared_ptr<PipelineElementWindow>  PipelineElementWindow::GetRightOutput()
{
    return _pRightOutput.lock();
}

std::shared_ptr<PipelineElementWindow>  PipelineElementWindow::GetBottomOutput()
{
    return _pBottomOutput.lock();
}

void PipelineElementWindow::SetRightOutput( std::shared_ptr<PipelineElementWindow> pElement )
{
    _pRightOutput = pElement;
}

void PipelineElementWindow::SetBottomOutput( std::shared_ptr<PipelineElementWindow> pElement )
{
    _pBottomOutput = pElement;
}

int PipelineElementWindow::GetInOutFlags()
{
    return _inOutFlags;
}

bool PipelineElementWindow::HasFreeInputs()
{
    const auto flags = GetInOutFlags();
    if ( flags & int( PEFlags_NoInput ) )
        return false;

    if ( ( flags & PEFlags_StrictlyOneInput ) && ( GetTopInput() || GetLeftInput() ) )
        return false;

    if ( ( flags & PEFlags_StrictlyTwoInputs ) && ( GetTopInput() && GetLeftInput() ) )
        return false;

    return true;
}

bool PipelineElementWindow::HasFreeOutputs()
{
    const auto flags = GetInOutFlags();
    if ( flags & PEFlags_NoOutput )
        return false;

    if ( ( flags & PEFlags_StrictlyOneOutput ) && ( GetBottomOutput() || GetRightOutput() ) )
        return false;

    return true;
}

size_t PipelineElementWindow::GetTaskCount()
{
    if ( _taskCount == 0 )
    {
        auto pPrimaryInput = GetLeftInput();
        if ( pPrimaryInput )
            _taskCount = pPrimaryInput->GetTaskCount();
    }

    return _taskCount;
}

void PipelineElementWindow::ResetTasks()
{
    _taskCount = 0;
    _completedTaskCount = 0;
    _taskReadiness = 0;
}

void PipelineElementWindow::DrawDialog()
{
    DrawPipelineElementControls();
    ImGui::SetCursorPosY( cElementHeight - ImGui::GetStyle().WindowPadding.y - ImGui::GetTextLineHeight() - 2 * ImGui::GetStyle().FramePadding.y );
    ImGui::ProgressBar( _taskCount > 0 ? ( float( _completedTaskCount ) + _taskReadiness ) / float( _taskCount ) : 0.0f, { _itemWidth, 0 } );
    ImGui::Dummy( { -1, 0 } );
}

ACMB_GUI_NAMESPACE_END
