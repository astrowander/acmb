#include "PipelineElementWindow.h"
#include "Serializer.h"
#include "MainWindow.h"

#include "./../Registrator/stacker.h"
#include "./../Cuda/CudaInfo.h"
#include "./../Cuda/CudaStacker.h"

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

std::expected<IBitmapPtr, std::string> PipelineElementWindow::RunTask( size_t i )
{
    if ( _completedTaskCount == 0 && (_inOutFlags & PEFlags_StrictlyTwoInputs) )
    {
        auto temp = ProcessSecondaryInput();
        if ( !temp.has_value() )
            return std::unexpected( temp.error() );

        _pSecondaryInputResult = temp.value();
    }

    const auto pPrimaryInput = GetPrimaryInput();
    if ( !pPrimaryInput )
        return std::unexpected( "No input " );

    auto relationType = (pPrimaryInput == GetLeftInput()) ? _leftInput.relationType : _topInput.relationType;
    if ( relationType == RelationType::Join )
    {
        auto pBitmap = pPrimaryInput->RunTaskAndReportProgress( 0 );
        if ( !pBitmap )
            return std::unexpected( pBitmap.error() );

        std::shared_ptr<BaseStacker> pStacker = cuda::isCudaAvailable() ? std::shared_ptr<BaseStacker>( new cuda::Stacker( **pBitmap, StackMode::DarkOrFlat ) ) :
            std::shared_ptr<BaseStacker>( new Stacker( **pBitmap, StackMode::DarkOrFlat ) );

        const size_t inputTaskCount = pPrimaryInput->GetTaskCount();
        if ( inputTaskCount == 0 )
            return std::unexpected( "No input frames" );

        try
        {
            for ( size_t i = 1; i < inputTaskCount; ++i )
            {
                pBitmap = pPrimaryInput->RunTaskAndReportProgress( i );
                if ( !pBitmap )
                    return std::unexpected( pBitmap.error() );

                pStacker->AddBitmap( *pBitmap );

                _taskReadiness = float( i ) / (inputTaskCount + 1);
            }

            const auto res = ProcessBitmapFromPrimaryInput( pStacker->GetResult() );
            _completedTaskCount = 1;
            _taskReadiness = 0.0f;
            return res;
        }
        catch ( std::exception& e )
        {
            return std::unexpected( e.what() );
        }
    }

    const auto taskRes = pPrimaryInput->RunTaskAndReportProgress( i );
    if ( !taskRes.has_value() )
        return std::unexpected( taskRes.error() );

    try
    {
        return ProcessBitmapFromPrimaryInput( taskRes.value(), i );
    }
    catch ( std::exception& e )
    {
        return std::unexpected( e.what() );
    }

}

std::expected<IBitmapPtr, std::string> PipelineElementWindow::ProcessSecondaryInput()
{
    if ( !(_inOutFlags & PEFlags_StrictlyTwoInputs) )
        return nullptr;

    auto pSecondaryInput = GetSecondaryInput();
    if ( !pSecondaryInput )
        return std::unexpected( "No secondary input" );

    auto relationType = (pSecondaryInput == GetLeftInput()) ? _leftInput.relationType : _topInput.relationType;
    if ( relationType == RelationType::Join )
    {
        auto pBitmap = pSecondaryInput->RunTaskAndReportProgress( 0 );
        if ( !pBitmap )
            return std::unexpected( pBitmap.error() );

        std::shared_ptr<BaseStacker> pStacker = cuda::isCudaAvailable() ? std::shared_ptr<BaseStacker>( new cuda::Stacker( **pBitmap, StackMode::DarkOrFlat ) ) :
            std::shared_ptr<BaseStacker>( new Stacker( **pBitmap, StackMode::DarkOrFlat ) );

        const size_t inputTaskCount = pSecondaryInput->GetTaskCount();
        if ( inputTaskCount == 0 )
            return std::unexpected( "No input frames" );

        try
        {
            for ( size_t i = 1; i < inputTaskCount; ++i )
            {
                pBitmap = pSecondaryInput->RunTaskAndReportProgress( i );
                if ( !pBitmap )
                    return std::unexpected( pBitmap.error() );

                pStacker->AddBitmap( *pBitmap );

                _taskReadiness = float( i ) / (inputTaskCount + 1);
            }

            const auto res = pStacker->GetResult();
            _completedTaskCount = 1;
            _taskReadiness = 0.0f;
            return res;
        }
        catch ( std::exception& e )
        {
            return std::unexpected( e.what() );
        }
    }

    const auto taskRes = pSecondaryInput->RunTaskAndReportProgress( 0 );
    if ( !taskRes.has_value() )
        return std::unexpected( taskRes.error() );

    return taskRes.value();
}

std::shared_ptr<PipelineElementWindow>  PipelineElementWindow::GetPrimaryInput()
{
    if ( _inOutFlags & PEFlags_NoInput )
        return nullptr;

    if ( _inOutFlags & PEFlags_StrictlyOneInput )
    {
        std::shared_ptr<PipelineElementWindow> res = GetLeftInput();
        if ( !res )
            res = GetTopInput();

        return res;
    }

    return _primaryInputIsOnLeft ? GetLeftInput() : GetTopInput();
}

std::shared_ptr<PipelineElementWindow>  PipelineElementWindow::GetSecondaryInput()
{
    if ( ! (_inOutFlags & PEFlags_StrictlyTwoInputs ) )
        return nullptr;

    return _primaryInputIsOnLeft ? GetTopInput() : GetLeftInput();
}

std::shared_ptr<PipelineElementWindow>  PipelineElementWindow::GetLeftInput()
{
    return _leftInput.pElement.lock();
}

std::shared_ptr<PipelineElementWindow>  PipelineElementWindow::GetTopInput()
{
    return _topInput.pElement.lock();
}

void PipelineElementWindow::SetLeftInput( std::shared_ptr<PipelineElementWindow> pLeftInput )
{
    _leftInput.pElement = pLeftInput;
    _serializedInputs.left = pLeftInput ? _leftInput.relationType : RelationType::None;
}

void PipelineElementWindow::SetTopInput( std::shared_ptr<PipelineElementWindow> pTopInput )
{
    _topInput.pElement = pTopInput;
    _serializedInputs.top = pTopInput ? _topInput.relationType : RelationType::None;
}

std::shared_ptr<PipelineElementWindow>  PipelineElementWindow::GetRightOutput()
{
    return _rightOutput.pElement.lock();
}

std::shared_ptr<PipelineElementWindow>  PipelineElementWindow::GetBottomOutput()
{
    return _bottomOutput.pElement.lock();
}

void PipelineElementWindow::SetRightOutput( std::shared_ptr<PipelineElementWindow> pElement )
{
    _rightOutput.pElement = pElement;
}

void PipelineElementWindow::SetBottomOutput( std::shared_ptr<PipelineElementWindow> pElement )
{
    _bottomOutput.pElement = pElement;
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
        auto pPrimaryInput = GetPrimaryInput();
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
        MainWindow::GetInstance( FontRegistry::Instance() ).LockInterface();
    }

    if ( ImGui::BeginPopup( "RenameElement" ) )
    {
        ImGui::InputText( "New name", _renameBuf.data(), _renameBuf.size() );

        if ( ImGui::IsKeyPressed( ImGuiKey_Enter ) )
        {
            const size_t length = strlen( _renameBuf.data() );
            if ( length > 0 )
                _name = std::string( _renameBuf.data(), length ) + "##R" + std::to_string( _gridPos.y ) + "C" + std::to_string( _gridPos.x );

            MainWindow::GetInstance( FontRegistry::Instance() ).UnlockInterface();
            _openRenamePopup = false;
        }

        if ( ImGui::IsKeyPressed( ImGuiKey_Escape ) )
        {
            MainWindow::GetInstance( FontRegistry::Instance() ).UnlockInterface();
            _openRenamePopup = false;
        }

        ImGui::EndPopup();
    }
}

void PipelineElementWindow::Serialize( std::ostream& out )
{
    gui::Serialize( GetSerializedStringSize(), out );
    gui::Serialize( _name, out );
    gui::Serialize( _serializedInputs, out );
    gui::Serialize( _primaryInputIsOnLeft, out );
}

void PipelineElementWindow::Deserialize( std::istream& in )
{
    _remainingBytes = sizeof( int );
    _remainingBytes = gui::Deserialize<int>( in, _remainingBytes );

    _name = acmb::gui::Deserialize<std::string>( in, _remainingBytes );
    _serializedInputs = gui::Deserialize<SerializedInputs>( in, _remainingBytes );
    _primaryInputIsOnLeft = gui::Deserialize<int>( in, _remainingBytes );
}

int PipelineElementWindow::GetSerializedStringSize()
{
    return gui::GetSerializedStringSize( _name ) + gui::GetSerializedStringSize( _serializedInputs ) + gui::GetSerializedStringSize( _primaryInputIsOnLeft );
}


ACMB_GUI_NAMESPACE_END
