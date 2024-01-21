#include "PipelineElementWindow.h"
#include "Serializer.h"
#include "MainWindow.h"
#include "ImGuiHelpers.h"

#include "./../Registrator/stacker.h"
#include "./../Cuda/CudaInfo.h"
#include "./../Cuda/CudaStacker.h"

#include "imgui/imgui_internal.h"

ACMB_GUI_NAMESPACE_BEGIN

PipelineElementWindow::PipelineElementWindow( const std::string& name, const Point& gridPos, int inOutFlags )
    : Window( name + "##R" + std::to_string( gridPos.y ) + "C" + std::to_string( gridPos.x ), { cElementWidth, cElementHeight } )
    , _itemWidth( cElementWidth - ImGui::GetStyle().WindowPadding.x * cMenuScaling )
    , _inOutFlags( inOutFlags )
    , _gridPos( gridPos )
    , _previewPopupName( name + "##Preview" )
{
}

Expected<IBitmapPtr, std::string> PipelineElementWindow::RunTaskAndReportProgress( size_t i )
{
    Expected<IBitmapPtr, std::string> res;
    try
    {
        res = RunTask( i );
    }
    catch ( std::exception& e )
    {
        ResetProgress( PropagationDir::Both );
        return unexpected( e.what() );
    }

    _completedTaskCount = res.has_value() ? i + 1 : 0;
    return res;
}

Expected<IBitmapPtr, std::string> PipelineElementWindow::RunTask( size_t i )
{
    const auto pPrimaryInput = GetPrimaryInput();
    if ( !pPrimaryInput )
        return unexpected( "Primary input of the '" + _name + "' element is not set" );

    const size_t primaryInputTaskCount = pPrimaryInput->GetTaskCount();
    if ( primaryInputTaskCount == 0 )
        return unexpected( "No input frames for the'" + _name + "' element" );

    if ( _completedTaskCount == 0 && (_inOutFlags & PEFlags_StrictlyTwoInputs) )
    {
        auto temp = ProcessSecondaryInput();
        if ( !temp.has_value() )
            return unexpected( temp.error() );

        if ( temp.value() )
            _pSecondaryInputResult = temp.value();
    }

    auto relationType = (pPrimaryInput == GetLeftInput()) ? _leftInput.relationType : _topInput.relationType;
    if ( relationType == RelationType::Join )
    {
        auto pBitmap = pPrimaryInput->RunTaskAndReportProgress( 0 );
        if ( !pBitmap )
            return unexpected( pBitmap.error() );

        std::shared_ptr<BaseStacker> pStacker = cuda::isCudaAvailable() ? std::shared_ptr<BaseStacker>( new cuda::Stacker( **pBitmap, StackMode::DarkOrFlat ) ) :
            std::shared_ptr<BaseStacker>( new Stacker( **pBitmap, StackMode::DarkOrFlat ) );

        try
        {
            for ( size_t i = 1; i < primaryInputTaskCount; ++i )
            {
                pBitmap = pPrimaryInput->RunTaskAndReportProgress( i );
                if ( !pBitmap )
                    return unexpected( pBitmap.error() );

                pStacker->AddBitmap( *pBitmap );

                _taskReadiness = float( i ) / (primaryInputTaskCount + 1);
            }

            const auto res = ProcessBitmapFromPrimaryInput( pStacker->GetResult() );
            _completedTaskCount = 1;
            _taskReadiness = 0.0f;
            return res;
        }
        catch ( std::exception& e )
        {
            return unexpected( e.what() );
        }
    }

    const auto taskRes = pPrimaryInput->RunTaskAndReportProgress( i );
    if ( !taskRes.has_value() )
        return unexpected( taskRes.error() );

    try
    {
        return ProcessBitmapFromPrimaryInput( taskRes.value(), i );
    }
    catch ( std::exception& e )
    {
        return unexpected( e.what() );
    }

}

Expected<IBitmapPtr, std::string> PipelineElementWindow::ProcessSecondaryInput()
{
    if ( !(_inOutFlags & PEFlags_StrictlyTwoInputs) )
        return nullptr;

    auto pSecondaryInput = GetSecondaryInput();
    if ( !pSecondaryInput )
        return unexpected( "Secondary input of the '" + _name + "' element is not set" );

    if ( pSecondaryInput->GetCompletedTaskCount() > 0 )
        return nullptr;

    auto relationType = (pSecondaryInput == GetLeftInput()) ? _leftInput.relationType : _topInput.relationType;
    if ( relationType == RelationType::Join )
    {
        auto pBitmap = pSecondaryInput->RunTaskAndReportProgress( 0 );
        if ( !pBitmap )
            return unexpected( pBitmap.error() );

        std::shared_ptr<BaseStacker> pStacker = cuda::isCudaAvailable() ? std::shared_ptr<BaseStacker>( new cuda::Stacker( **pBitmap, StackMode::DarkOrFlat ) ) :
            std::shared_ptr<BaseStacker>( new Stacker( **pBitmap, StackMode::DarkOrFlat ) );

        const size_t inputTaskCount = pSecondaryInput->GetTaskCount();
        if ( inputTaskCount == 0 )
            return unexpected( "No input frames for the'" + _name + "' element" );

        try
        {
            for ( size_t i = 1; i < inputTaskCount; ++i )
            {
                pBitmap = pSecondaryInput->RunTaskAndReportProgress( i );
                if ( !pBitmap )
                    return unexpected( pBitmap.error() );

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
            return unexpected( e.what() );
        }
    }

    const auto taskRes = pSecondaryInput->RunTaskAndReportProgress( 0 );
    if ( !taskRes.has_value() )
        return unexpected( taskRes.error() );

    return taskRes.value();
}

std::shared_ptr<PipelineElementWindow>  PipelineElementWindow::GetPrimaryInput() const
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

    return _primaryInputIsOnTop ? GetTopInput() : GetLeftInput();
}

std::shared_ptr<PipelineElementWindow>  PipelineElementWindow::GetSecondaryInput() const
{
    if ( ! (_inOutFlags & PEFlags_StrictlyTwoInputs ) )
        return nullptr;

    return _primaryInputIsOnTop ? GetLeftInput() : GetTopInput();
}

std::shared_ptr<PipelineElementWindow>  PipelineElementWindow::GetLeftInput() const
{
    return _leftInput.pElement.lock();
}

std::shared_ptr<PipelineElementWindow>  PipelineElementWindow::GetTopInput() const
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

std::shared_ptr<PipelineElementWindow>  PipelineElementWindow::GetRightOutput() const
{
    return _rightOutput.pElement.lock();
}

std::shared_ptr<PipelineElementWindow>  PipelineElementWindow::GetBottomOutput() const
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

int PipelineElementWindow::GetInOutFlags() const
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

size_t PipelineElementWindow::GetCompletedTaskCount()
{
    return _completedTaskCount;
}

void PipelineElementWindow::ResetTasks()
{
    _taskCount = 0;
    _completedTaskCount = 0;
    _taskReadiness = 0;
}

void PipelineElementWindow::ResetProgress( PropagationDir dir )
{
    ResetTasks();

    if ( int( dir ) & int( PropagationDir::Backward ) )
    {
        auto pPrimaryInput = GetPrimaryInput();
        if ( pPrimaryInput )
            pPrimaryInput->ResetProgress( dir );
    }
    
    if ( int( dir ) & int( PropagationDir::Forward ) )
    {
        auto pOutput = GetRightOutput();
        if ( !pOutput )
            pOutput = GetBottomOutput();
        
        if ( pOutput )
        {
            //if ( pOutput->GetPrimaryInput().get() == this && std::dynamic_pointer_cast< StackerWindow >(pOutput) == nullptr )
                //pOutput->_taskCount = _taskCount;

            pOutput->ResetProgress( dir );
        }
    }
}

bool PipelineElementWindow::DrawHeader()
{
    if ( !Window::DrawHeader() )
        return false;

    auto window = ImGui::GetCurrentWindow();
    constexpr float titleBarHeight = 24.0f;

    const auto oldCursorPos = ImGui::GetCursorPos();
    const ImVec2 topLeft{ window->Pos.x + 1, window->Pos.y + 1 };
    const ImVec2 bottomRight{ topLeft.x + _size.x - 2, topLeft.y + titleBarHeight - 2 };
    
    auto drawList = ImGui::GetWindowDrawList();
    ImGui::PushClipRect( topLeft, bottomRight, false );
    drawList->AddRectFilled( topLeft, bottomRight, ImGui::GetColorU32( ImGuiCol_TitleBgActive ) );
    
    ImGui::SetCursorPosY( oldCursorPos.y - ImGui::GetStyle().WindowPadding.y * 0.5f );
    ImGui::Text( "%s", _name.substr(0, _name.find_first_of('#') ).c_str());
    ImGui::SameLine();

    constexpr float previewButtonWidth = titleBarHeight;
    constexpr float previewButtonHeight = titleBarHeight;
    ImGui::SetCursorPos( { window->Size.x - previewButtonWidth, 0.0f } );

    ImGui::PushStyleColor( ImGuiCol_Button, { 0.0f, 1.0f, 0.0f, 0.4f } );
    ImGui::PushFont( FontRegistry::Instance().iconsSmall );
    ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { 0, 0 });

    UI::Button( "\xef\x80\xbe", { previewButtonWidth, previewButtonHeight }, [&]
    {
        if ( !_pPreviewTexture )
        {
            auto previewExp = GeneratePreviewTexture();
            if ( !previewExp.has_value() )
            {
                UI::ShowModalMessage( { "Unable to generate preview" }, UI::ModalMessageType::Error, _showError = true );
                return;
            }            
        }

        if ( _pPreviewTexture )
            _showPreview = true;
    }, "Show preview of the image processed by this tool" );

    ImGui::PopStyleVar();
    ImGui::PopFont();
    ImGui::PopStyleColor();

    ImGui::PopClipRect();
    ImGui::SetCursorPosY( oldCursorPos.y + titleBarHeight );
    return true;
}

void PipelineElementWindow::DrawDialog()
{
    ImGui::PushItemWidth( 50.0f * cMenuScaling );
    DrawPipelineElementControls();
    ImGui::PopItemWidth();

    ImGui::SetCursorPosY( cElementHeight - ImGui::GetStyle().WindowPadding.y - ImGui::GetTextLineHeight() - 2 * ImGui::GetStyle().FramePadding.y );
    const auto taskCount = GetTaskCount();
    ImGui::ProgressBar( taskCount > 0 ? ( float( _completedTaskCount ) + _taskReadiness ) / float( taskCount ) : 0.0f, { _itemWidth, 0 } );

    auto& mainWindow = MainWindow::GetInstance( FontRegistry::Instance() );

    if ( !mainWindow.IsInterfaceLocked() )
    {
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
            mainWindow.LockInterface();
        }

        if ( ImGui::BeginPopup( "RenameElement" ) )
        {
            ImGui::InputText( "New name", _renameBuf.data(), _renameBuf.size() );

            if ( ImGui::IsKeyPressed( ImGuiKey_Enter ) )
            {
                const size_t length = strlen( _renameBuf.data() );
                if ( length > 0 )
                    _name = std::string( _renameBuf.data(), length ) + "##R" + std::to_string( _gridPos.y ) + "C" + std::to_string( _gridPos.x );

                mainWindow.UnlockInterface();
                _openRenamePopup = false;
            }

            if ( ImGui::IsKeyPressed( ImGuiKey_Escape ) )
            {
                mainWindow.UnlockInterface();
                _openRenamePopup = false;
            }

            ImGui::EndPopup();
        }

        if ( _showPreview && !ImGui::IsPopupOpen( _previewPopupName.c_str() ) )
        {
            ImGui::OpenPopup( _previewPopupName.c_str() );
            const auto mainWindow = ImGui::FindWindowByName( "acmb" );
            ImVec2 previewPos { std::max( mainWindow->Size.x - 1280.0f, 0.0f ), 0.0f };
            ImGui::SetNextWindowPos( previewPos );
        }

        if ( ImGui::BeginPopup( _previewPopupName.c_str(), ImGuiWindowFlags_NoFocusOnAppearing ) )
        {
            if ( !_pPreviewTexture )
            {
                auto previewExp = GeneratePreviewTexture();
                if ( !previewExp.has_value() )
                    UI::ShowModalMessage( { "Unable to generate preview" }, UI::ModalMessageType::Error, _showError = true );
            }

            ImGui::Image( _pPreviewTexture->GetTexture(), { float( _pPreviewTexture->GetWidth() ), float( _pPreviewTexture->GetHeight() ) } );
            if ( ImGui::IsKeyPressed( ImGuiKey_Escape ) )
            {
                ImGui::CloseCurrentPopup();
                _showPreview = false;
            }
            ImGui::EndPopup();
        }

        if ( _showError )
            UI::ShowModalMessage( { _error }, UI::ModalMessageType::Error, _showError );
    }
}

void PipelineElementWindow::Serialize( std::ostream& out ) const
{
    gui::Serialize( GetSerializedStringSize(), out );
    gui::Serialize( _name, out );
    gui::Serialize( _serializedInputs, out );
    gui::Serialize( _primaryInputIsOnTop, out );
}

void PipelineElementWindow::Deserialize( std::istream& in )
{
    _remainingBytes = sizeof( int );
    _remainingBytes = gui::Deserialize<int>( in, _remainingBytes );

    _name = acmb::gui::Deserialize<std::string>( in, _remainingBytes );
    _serializedInputs = gui::Deserialize<SerializedInputs>( in, _remainingBytes );
    _primaryInputIsOnTop = gui::Deserialize<bool>( in, _remainingBytes );
}

int PipelineElementWindow::GetSerializedStringSize() const
{
    return gui::GetSerializedStringSize( _name ) + gui::GetSerializedStringSize( _serializedInputs ) + gui::GetSerializedStringSize( _primaryInputIsOnTop );
}

Expected<void, std::string> PipelineElementWindow::GeneratePreviewTexture()
{
    auto pPrimaryInput = GetPrimaryInput();
    auto pSecondaryInput = GetSecondaryInput();
    if ( !(_inOutFlags | PEFlags_NoInput) && (!pPrimaryInput || pPrimaryInput->GetTaskCount() == 0) )
        return unexpected( "no primary input element" );

    if ( !(_inOutFlags | PEFlags_StrictlyTwoInputs) && (!pSecondaryInput || pSecondaryInput->GetTaskCount() == 0) )
        return unexpected( "no secondary input element" );

    if ( pPrimaryInput && !pPrimaryInput->GetPreviewBitmap() )
        pPrimaryInput->GeneratePreviewTexture();

    if ( pSecondaryInput && !pSecondaryInput->GetPreviewBitmap() )
        pSecondaryInput->GeneratePreviewTexture();

    try
    {
        if ( auto res = GeneratePreviewBitmap(); !res )
            return unexpected( res.error() );

        _pPreviewTexture = std::make_unique<Texture>( _pPreviewBitmap );
        return {};
    }
    catch ( std::exception& e )
    {
        return unexpected( e.what() );
    }
}

void PipelineElementWindow::ResetPreview()
{
    _pPreviewBitmap.reset();
    _pPreviewTexture.reset();
    auto output = GetRightOutput();
    if ( !output )
        output = GetBottomOutput();

    if ( output )
        output->ResetPreview();
}

Expected<Size, std::string> PipelineElementWindow::GetBitmapSize()
{
    auto pPrimaryInput = GetPrimaryInput();
    if ( !pPrimaryInput )
        return unexpected( "no primary input element" );

    return pPrimaryInput->GetBitmapSize();
}

ACMB_GUI_NAMESPACE_END
