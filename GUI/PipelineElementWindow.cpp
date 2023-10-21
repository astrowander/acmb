#include "PipelineElementWindow.h"
#include "Serializer.h"

ACMB_GUI_NAMESPACE_BEGIN

PipelineElementWindow::PipelineElementWindow( const std::string& name, const Point& gridPos, int inOutFlags )
    : Window( name + "##R" + std::to_string( gridPos.y ) + "C" + std::to_string( gridPos.x ), { cElementWidth, cElementHeight } )
    , _inOutFlags( inOutFlags )
    , _itemWidth( cElementWidth - ImGui::GetStyle().WindowPadding.x * cMenuScaling )
    , _gridPos( gridPos )
{
}

std::expected<IBitmapPtr, std::string> PipelineElementWindow::RunTaskAndReportProgress( size_t i )
{
    std::expected<IBitmapPtr, std::string> res;
    try
    {
        res = RunTask( i );
    }
    catch ( std::exception& e )
    {
        res = std::unexpected( e.what() );
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
    pPrimaryInput ? ( _actualInputs |= 1 ) : ( _actualInputs &= 2 );
}

void PipelineElementWindow::SetTopInput( std::shared_ptr<PipelineElementWindow> pSecondaryInput )
{
    _pTopInput = pSecondaryInput;
    pSecondaryInput ? ( _actualInputs |= 2 ) : ( _actualInputs &= 1 );
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
    ImGui::PushItemWidth( 50.0f * cMenuScaling );
    DrawPipelineElementControls();
    ImGui::PopItemWidth();

    ImGui::SetCursorPosY( cElementHeight - ImGui::GetStyle().WindowPadding.y - ImGui::GetTextLineHeight() - 2 * ImGui::GetStyle().FramePadding.y );
    ImGui::ProgressBar( _taskCount > 0 ? ( float( _completedTaskCount ) + _taskReadiness ) / float( _taskCount ) : 0.0f, { _itemWidth, 0 } );

    if ( ImGui::IsMouseDoubleClicked( ImGuiMouseButton_Left ) )
    {
        auto mousePos = ImGui::GetMousePos();
        const auto windowPos = ImGui::GetWindowPos();
        mousePos.x -= windowPos.x;
        mousePos.y -= windowPos.y;

        const auto& style = ImGui::GetStyle();
        const float titleHeight = style.FramePadding.y * 2 + ImGui::GetTextLineHeight();

        if ( mousePos.y >= 0 && mousePos.y < titleHeight && mousePos.x >= 0 && mousePos.x <= ImGui::GetWindowSize().x )
        {
            _openRenamePopup = true;
        }
    }

    if ( _openRenamePopup )
    {
        ImGui::OpenPopup( "RenameElement" );
    }

    if ( ImGui::BeginPopup( "RenameElement" ) )
    {
        ImGui::InputText( "New name", _renameBuf.data(), _renameBuf.size() );

        if ( ImGui::IsKeyPressed( ImGuiKey_Enter ) )
        {
            _name = std::string( _renameBuf.data(), strlen( _renameBuf.data() ) ) + "##R" + std::to_string( _gridPos.y ) + "C" + std::to_string( _gridPos.x );
            _openRenamePopup = false;
        }

        ImGui::EndPopup();
    }
}

void PipelineElementWindow::Serialize( std::ostream& out )
{
    gui::Serialize( _name, out );
    gui::Serialize( _actualInputs, out );
}

void PipelineElementWindow::Deserialize( std::istream& in )
{
    _name = acmb::gui::Deserialize<std::string>( in );
    _actualInputs = gui::Deserialize<char>( in );
}

ACMB_GUI_NAMESPACE_END
