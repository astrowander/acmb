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

    auto pInputBitmapOrErr = pInput->GeneratePreviewBitmap(forNextElement, fullSize);
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
    OnInputChanged();
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
    
    ImGui::PopClipRect();
    ImGui::SetCursorPosY( oldCursorPos.y + titleBarHeight );
    return true;
}

int PipelineElementWindow::GetElementsCount() const
{
    int count = 1;
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

Expected<void, std::string> PipelineElementWindow::FinalizePreviewBitmap(bool forNextElement, bool fullSize)
{
    if ( !_pPreviewBitmap.load(std::memory_order_acquire) )
    {
        _previewWorkers.emplace_back();
        auto self = std::static_pointer_cast<PipelineElementWindow>(shared_from_this());
        _previewWorkers.back().Start( [self, forNextElement, fullSize]( auto reportProgress )
        {
            auto pPreviewOrErr = self->GeneratePreviewBitmap(forNextElement, fullSize);
            if ( !pPreviewOrErr.has_value() || !reportProgress(0.5f) )
            {
                // Handle error (e.g., log it, set an error state, etc.)
                return;
            }

            auto targetSizeOrErr = fullSize ? self->GetBitmapSize() : Size{ int(MainWindow::GetInstance().GetImageRegionAvail().width), int(MainWindow::GetInstance().GetImageRegionAvail().height) };
            if ( !targetSizeOrErr )
            {
                // Handle error
                return;
            }
            const auto targetSize = targetSizeOrErr.value();
            auto pPreview = pPreviewOrErr.value();
            if ( !pPreview || !reportProgress(0.55f) )
                return;

            Size srcSize{ int(pPreview->GetWidth()),int(pPreview->GetHeight()) };
            IBitmapPtr pResizedPreview;
            if ( targetSize != srcSize )
            {
                pResizedPreview = ResizeTransform::Resize(pPreview, ResizeTransform::GetSizeWithPreservedRatio(srcSize, targetSize));
            }
            else
            {
                pResizedPreview = pPreview;
            }
            
            if ( !reportProgress(1.0f) )
                return;

            self->_pPreviewBitmap.store(pResizedPreview, std::memory_order_release);
        });
    }

    return {};
}

Expected<std::shared_ptr<Texture>, std::string> PipelineElementWindow::GetPreviewTexture()
{
    if ( !_pPreviewBitmap.load(std::memory_order_acquire) )
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
        for ( auto it = _previewWorkers.begin(); it != _previewWorkers.end(); )
        {
            const auto status = it->GetStatus();
            if ( status != AsyncWorker::Status::Idle && status != AsyncWorker::Status::Running )
            {
                it = _previewWorkers.erase( it );
            }
            else
            {
                ++it;
            }
        }

        auto resOrErr = FinalizePreviewBitmap(false, false);
        if ( !resOrErr.has_value() )
        {
            return unexpected(resOrErr.error());
        }

        auto pPreviewBitmap = _pPreviewBitmap.load(std::memory_order_acquire);
        if ( pPreviewBitmap )
        {
            _pPreviewTexture = std::make_unique<Texture>(pPreviewBitmap);
        }

        return {};
    }
    catch ( std::exception& e )
    {
        return unexpected( e.what() );
    }
}

void PipelineElementWindow::ResetPreview( PropagationDir dir )
{
    // Cancel all preview workers so they stop as soon as possible
    for ( auto& worker : _previewWorkers )
        worker.Cancel();

    _pPreviewBitmap.store(nullptr);
    //_pPreviewTexture.reset();

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
