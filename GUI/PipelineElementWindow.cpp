#include "PipelineElementWindow.h"
#include "Serializer.h"
#include "MainWindow.h"
#include "ImGuiHelpers.h"
#include "SettingsInterpolationUser.h"

#include "./../Registrator/stacker.h"
#include "./../Cuda/CudaInfo.h"
#include "./../Cuda/CudaStacker.h"
#include "./../Transforms/ResizeTransform.h"

#include "imgui/imgui_internal.h"

#include <future>

static int uniqueId = 0;
ACMB_GUI_NAMESPACE_BEGIN

PipelineElementWindow::PipelineElementWindow(const std::string& name)
    : Window( name + "##C" + std::to_string(uniqueId++), { cElementWidth, cElementHeight } )
    , _itemWidth( cElementWidth - ImGui::GetStyle().WindowPadding.x * cMenuScaling )
{
}

Expected<IBitmapPtr, std::string> PipelineElementWindow::RunTaskAndReportProgress( size_t i )
{
    Expected<IBitmapPtr, std::string> res;
    try
    {
        res = RunTask( i );
        if ( !MainWindow::GetInstance().IsInterfaceLocked() )
            throw std::runtime_error( "Processing was interrupted" );
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
    const auto pPrimaryInput = GetInput();
    if ( !pPrimaryInput )
        return unexpected( "Primary input of the '" + _name + "' element is not set" );

    const size_t primaryInputTaskCount = pPrimaryInput->GetTaskCount();
    if ( primaryInputTaskCount == 0 )
        return unexpected( "No input frames for the'" + _name + "' element" );

    const auto taskRes = pPrimaryInput->RunTaskAndReportProgress( i );
    if ( !taskRes.has_value() )
        return unexpected( taskRes.error() );

    try
    {
        auto res = ProcessBitmapFromPrimaryInput( taskRes.value(), i );
        if ( !_error.empty() )
            return unexpected( _error );

        return res;
    }
    catch ( std::exception& e )
    {
        return unexpected( e.what() );
    }

}

Expected<IBitmapPtr, std::string> PipelineElementWindow::GetInputPreview(bool forNextElement, bool fullSize) const
{
    auto pInput = GetInput();
    if ( !pInput )
        return unexpected("Primary input of the '" + _name + "' element is not set");

    auto pInputBitmapOrErr = pInput->GetPreviewBitmap(forNextElement, fullSize);
    if ( !pInputBitmapOrErr )
        return unexpected(pInputBitmapOrErr.error());

    return pInputBitmapOrErr.value()->Clone();
}

std::shared_ptr<PipelineElementWindow>  PipelineElementWindow::GetInput() const
{
    return _input.lock();
}

void PipelineElementWindow::SetInput( std::shared_ptr<PipelineElementWindow> pLeftInput )
{
    _input = pLeftInput;
}

std::shared_ptr<PipelineElementWindow>  PipelineElementWindow::GetOutput() const
{
    return _output;
}

void PipelineElementWindow::SetOutput( std::shared_ptr<PipelineElementWindow> pElement )
{
    _output = pElement;
}

int PipelineElementWindow::GetFollowingElementsCount() const
{
    int count = 1;
    auto pOutput = this;
    while ( pOutput = pOutput->GetOutput().get() )
    {
        ++count;
    }
    return count;
}

size_t PipelineElementWindow::GetTaskCount(bool update)
{
    if ( update || _taskCount == 0 )
    {
        auto pPrimaryInput = GetInput();
        if ( pPrimaryInput )
        {
            _taskCount = pPrimaryInput->GetTaskCount(update);
            if ( auto pSettingsInterpolationUser = dynamic_cast<ISettingsInterpolationUser*>(this); update && pSettingsInterpolationUser != nullptr)
            {
                pSettingsInterpolationUser->CutExtraKeyframes( std::max ( int( _taskCount ), 1 ) );
            }
        }
    }

    return _taskCount;
}

size_t PipelineElementWindow::GetCompletedTaskCount()
{
    return _completedTaskCount;
}

void PipelineElementWindow::ResetTasks()
{
    _completedTaskCount = 0;
    _taskReadiness = 0;
}

void PipelineElementWindow::ResetProgress( PropagationDir dir )
{
    ResetTasks();

    if ( int( dir ) & int( PropagationDir::Backward ) )
    {
        auto pPrimaryInput = GetInput();
        if ( pPrimaryInput && pPrimaryInput->GetCompletedTaskCount() > 0 )
            pPrimaryInput->ResetProgress( dir );
    }
    
    if ( int( dir ) & int( PropagationDir::Forward ) )
    {        
        if ( auto pOutput = GetOutput() )
        {
            //if ( pOutput->GetInput().get() == this && std::dynamic_pointer_cast< StackerWindow >(pOutput) == nullptr )
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

    const bool isPreviewOpen = ImGui::IsPopupOpen( ImGuiID( 0 ), ImGuiPopupFlags_AnyPopupId );
    UI::Button( "\xef\x80\xbe", { previewButtonWidth, previewButtonHeight }, [&]
    {
        if ( isPreviewOpen )
            return;

        if ( !_pPreviewTexture )
        {
            auto previewExp = GeneratePreviewTexture();
            if ( !previewExp.has_value() )
            {
                _error = previewExp.error();
                _showError = true;
                //UI::ShowModalMessage( { _error }, UI::ModalMessageType::Error, _showError = true );
                return;
            }            
        }

        if ( _pPreviewTexture )
            _showPreview = true;
    }, isPreviewOpen ? "Another preview is already opened" : "Show preview of the image processed by this tool" );

    ImGui::PopStyleVar();
    ImGui::PopFont();
    ImGui::PopStyleColor();

    ImGui::PopClipRect();
    ImGui::SetCursorPosY( oldCursorPos.y + titleBarHeight );
    return true;
}

size_t PipelineElementWindow::GetElementsCount() const
{
    size_t count = 1;
    auto pOutput = this;
    while ( pOutput = pOutput->GetOutput().get() )
    {
        ++count;
    }
    return count;
}

void PipelineElementWindow::DrawDialog()
{
    const auto taskCount = GetTaskCount();
    ImGui::ProgressBar(taskCount > 0 ? (float(_completedTaskCount) + _taskReadiness) / float(taskCount) : 0.0f, { _itemWidth, 0 });

    ImGui::PushItemWidth( 50.0f * cMenuScaling );
    DrawPipelineElementControls();
    ImGui::PopItemWidth();

    auto& mainWindow = MainWindow::GetInstance();
    if ( !mainWindow.IsInterfaceLocked() && ImGui::IsMouseClicked( ImGuiMouseButton_Right ) )
    {
        auto mousePos = ImGui::GetMousePos();
        const auto windowPos = ImGui::GetWindowPos();
        mousePos.x -= windowPos.x;
        mousePos.y -= windowPos.y;

        const auto& style = ImGui::GetStyle();
        const float titleHeight = style.FramePadding.y * 2 + ImGui::GetTextLineHeight();

        if ( mousePos.y >= 0 && mousePos.y < titleHeight && mousePos.x >= 0 && mousePos.x <= ImGui::GetWindowSize().x )
        {
            ImGui::OpenPopup( "RenameElement" );
            mainWindow.LockInterface();
        }
    }

    if ( ImGui::BeginPopup( "RenameElement" ) )
    {
        ImGui::InputText( "New name", _renameBuf.data(), _renameBuf.size() );

        if ( ImGui::IsKeyPressed( ImGuiKey_Enter ) )
        {
            const size_t length = strlen( _renameBuf.data() );
            if ( length > 0 )
                _name = std::string( _renameBuf.data(), length ) +  "##C" + std::to_string( uniqueId++ );

            mainWindow.UnlockInterface();
            ImGui::CloseCurrentPopup();

        }

        if ( ImGui::IsKeyPressed( ImGuiKey_Escape ) )
        {
            mainWindow.UnlockInterface();
            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }

    if ( _showPreview && !ImGui::IsPopupOpen( cPreviewPopupName.c_str() ) && _pPreviewTexture )
    {
        ImGui::OpenPopup( cPreviewPopupName.c_str() );
        ImVec2 previewPos;
        if ( const auto mainWindow = ImGui::FindWindowByName( "acmb" ); ImGui::GetMousePos().x < mainWindow->Size.x / 2 )
            previewPos.x = mainWindow->Size.x - _pPreviewTexture->GetWidth();
        
        ImGui::SetNextWindowPos( previewPos );
    }

    if ( _showPreview && ImGui::BeginPopup( cPreviewPopupName.c_str(), ImGuiWindowFlags_NoFocusOnAppearing ) )
    {
        if ( !_pPreviewTexture )
        {
            auto previewExp = GeneratePreviewTexture();
            if ( !previewExp.has_value() )
            {
                _showPreview = false;
                ImGui::CloseCurrentPopup();
                _error = previewExp.error();
                _showError = true;                
            }
        }

        if ( _pPreviewTexture )            
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

void PipelineElementWindow::Serialize( std::ostream& out ) const
{
    gui::Serialize( GetSerializedStringSize(), out );
    gui::Serialize( _name, out );
}

bool PipelineElementWindow::Deserialize( std::istream& in )
{
    _remainingBytes = sizeof( int );
    _remainingBytes = gui::Deserialize<int>( in, _remainingBytes );

    auto savedName = gui::Deserialize<std::string>( in, _remainingBytes );
    if ( savedName.empty() || savedName.back() == '\0' )
        return false;
    
    _name = std::move( savedName );
    return true;
}

int PipelineElementWindow::GetSerializedStringSize() const
{
    return gui::GetSerializedStringSize(_name);
}

std::string PipelineElementWindow::GetTaskName(size_t taskNumber) const
{
    auto pPrimaryInput = GetInput();
    return pPrimaryInput ? pPrimaryInput->GetTaskName(taskNumber) : std::string{};
}

Expected<IBitmapPtr, std::string> PipelineElementWindow::GetPreviewBitmap(bool forNextElement, bool fullSize)
{
    if ( !_pPreviewBitmap.load() )
    {

        std::future< Expected<IBitmapPtr, std::string>> previewGenTask = std::async(std::launch::async, [this, forNextElement, fullSize]() -> Expected<IBitmapPtr, std::string>
        {
            if ( _isGeneratingPreviewCancelled.load() )
                return unexpected("preview generation was cancelled");

            auto pPreviewOrErr = GeneratePreviewBitmap(forNextElement, fullSize);
            if ( !pPreviewOrErr.has_value() )
                return unexpected(pPreviewOrErr.error());

            if ( _isGeneratingPreviewCancelled.load() )
                return unexpected("preview generation was cancelled");

            auto targetSizeOrErr = fullSize ? GetBitmapSize() : Size{ int(MainWindow::GetInstance().GetImageRegionAvail().width), int(MainWindow::GetInstance().GetImageRegionAvail().height) };
            if ( !targetSizeOrErr )
                return unexpected(targetSizeOrErr.error());
            const auto targetSize = targetSizeOrErr.value();

            

            auto pPreview = pPreviewOrErr.value();

            if ( _isGeneratingPreviewCancelled.load() || !pPreview )
                return unexpected("preview generation was cancelled");

            Size srcSize{ int(pPreview->GetWidth()),int(pPreview->GetHeight()) };
            if ( targetSize != srcSize )
            {
                return ResizeTransform::Resize(pPreview, ResizeTransform::GetSizeWithPreservedRatio(srcSize, targetSize));
            }
            else
            {
                return pPreview;
            }
        });

        previewGenTask.wait();
        auto previewRes = previewGenTask.get();
        if ( !previewRes.has_value() )
            return unexpected(previewRes.error());

        _pPreviewBitmap.store(previewRes.value());
    }

    return _pPreviewBitmap.load();
}

Expected<std::shared_ptr<Texture>, std::string> PipelineElementWindow::GetPreviewTexture()
{
    if ( !_pPreviewTexture )
    {
        auto res = GeneratePreviewTexture();
        if ( !res.has_value() )
            return unexpected( res.error() );
    }

    return _pPreviewTexture;
}

Expected<void, std::string> PipelineElementWindow::GeneratePreviewTexture()
{
    try
    {
        MainWindow::GetInstance().LockInterface();
        CancelPreviewGeneration(false);
        auto pPreviewBitmap = GetPreviewBitmap(false, false);
        if ( !pPreviewBitmap )
        {
            MainWindow::GetInstance().UnlockInterface();
            return unexpected(pPreviewBitmap.error());
        }
        
        _pPreviewTexture = std::make_unique<Texture>( _pPreviewBitmap );
        MainWindow::GetInstance().UnlockInterface();
        return {};
    }
    catch ( std::exception& e )
    {
        MainWindow::GetInstance().UnlockInterface();
        return unexpected( e.what() );
    }
}

void PipelineElementWindow::ResetPreview( PropagationDir dir )
{
    _pPreviewBitmap.store(nullptr);
    _pPreviewTexture.reset();

    if ( int(dir) & int(PropagationDir::Forward) )
    {
        if ( auto output = GetOutput() )
            output->ResetPreview(PropagationDir::Forward);
    }

    if ( int(dir) & int(PropagationDir::Backward) )
    {
        if ( auto input = GetInput() )
            input->ResetPreview(PropagationDir::Backward);
    }
}

Expected<Size, std::string> PipelineElementWindow::GetBitmapSize()
{
    auto pPrimaryInput = GetInput();
    if ( !pPrimaryInput )
        return unexpected( "no primary input element" );

    return pPrimaryInput->GetBitmapSize();
}

void PipelineElementWindow::OnPreviewedFrameNumberChanged(int val)
{
    if ( GetPreviewedFrameNumber() == val || val < 0 || val >= _taskCount )
        return;

    SetPreviewedFrameNumber(val);
    ResetPreview(PropagationDir::None);

    auto pInput = GetInput();
    while ( pInput )
    {
        pInput->OnPreviewedFrameNumberChanged(val);
        pInput->ResetPreview(PropagationDir::None);

        pInput = pInput->GetInput();
    }

    auto pOutput = GetOutput();
    while ( pOutput)
    {
        pOutput->ResetPreview(PropagationDir::None);
        pOutput->OnPreviewedFrameNumberChanged(val);

        pOutput = pOutput->GetOutput();
    }
}

ACMB_GUI_NAMESPACE_END
