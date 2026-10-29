#include "MainWindow.h"
#include "ImageReaderWindow.h"
#include "ImageWriterWindow.h"
#include "SettingsInterpolationUser.h"
#include "FontRegistry.h"
#include "FileDialog.h"
#include "ImGuiHelpers.h"
#include "./../Tools/SystemTools.h"
#include "./../Cuda/CudaInfo.h"
#include <fstream>
#include <filesystem>
#include <stdexcept>

#ifdef _WIN32
#include <atltypes.h>
#include <WinUser.h>
#elif defined ( __linux__ )
#include <GLFW/glfw3.h>
#endif

ACMB_GUI_NAMESPACE_BEGIN

static constexpr float cMenuButtonSize = 60.0f;
constexpr int cItemCountInRow = 5;

static constexpr float cHeadRowHeight = MainWindowLayout::HeaderHeight;
static constexpr float cMenuRowHeight = cMenuButtonSize + 30.0f;

static constexpr float cGridLeft = MainWindowLayout::GridLeft;
static constexpr float cGridCellPadding = 25.0f;

static constexpr float cGridCellMinWidth = PipelineElementWindow::cElementWidth + 2.0f * cGridCellPadding;
static constexpr float cGridCellHeight = PipelineElementWindow::cElementHeight + 2.0f * cGridCellPadding;
static_assert(cGridCellMinWidth == MainWindowLayout::CellMinWidth);
static_assert(cGridCellHeight == MainWindowLayout::CellHeight);

MainWindow::MainWindow( const ImVec2& pos, const ImVec2& size, const FontRegistry& fontRegistry )
    : Window( "acmb", size )
    , _fontRegistry( fontRegistry )
    , _gridCellSize( { cGridCellMinWidth, cGridCellHeight } )
{
    SetPos( pos );
    _visibleCellsCount = MainWindowLayout::CountVisibleCells(size.x);

    MenuItemsHolder::GetInstance().AddItem( "Run", 1, "\xef\x81\x8B", "Run", "Start processing", [this] (size_t, bool)
    {
        if ( _processingFuture.valid() )
            return;

        _errors.clear();
        if ( !AutoSaveProject() )
            return;
        _showResultsPopup = false;
        _stopRequested.store(false);
        _isBusy = true;
        LockInterface();
        _startTime = std::chrono::high_resolution_clock::now();
        _durationString.clear();

        try
        {
            _processingFuture = std::async(std::launch::async, [head = _pPipelineHead]
            {
                std::vector<std::string> errors;

                if ( head == nullptr )
                {
                    errors.push_back( "Pipeline is empty" );
                }

                if ( !std::dynamic_pointer_cast<ImageReaderWindow>(head) )
                {
                    errors.push_back( "First element must be an image reader" );
                }

                auto pTailElement = head;
                while ( pTailElement && pTailElement->GetOutput() != nullptr )
                {
                    pTailElement = pTailElement->GetOutput();
                    if ( std::dynamic_pointer_cast<ImageReaderWindow>(pTailElement) )
                    {
                        errors.push_back( "Image reader can only be the first element in the pipeline" );
                        break;
                    }

                    if ( std::dynamic_pointer_cast<ImageWriterWindow>(pTailElement) && pTailElement->GetOutput() != nullptr )
                    {
                        errors.push_back( "Image writer can only be the last element in the pipeline" );
                        break;
                    }
                }

                auto pWriter = std::dynamic_pointer_cast<ImageWriterWindow>(pTailElement);
                if ( !pWriter )
                {
                    errors.push_back( "Last element must be an image writer" );
                }

                if ( errors.empty() )
                {
                    pWriter->ResetTasks();
                    return pWriter->ExportAllImages();
                }

                return errors;
            } );
        }
        catch ( const std::exception& e )
        {
            _errors.push_back(e.what());
            FinishProcessing();
        }
        catch ( ... )
        {
            _errors.push_back("Unknown error while starting processing");
            FinishProcessing();
        }
    } );

    MenuItemsHolder::GetInstance().AddItem( "Run", 2, "\xef\x81\x8D", "Stop", "Stop processing", [this] (size_t, bool)
    {
        if ( _isBusy )
            _stopRequested.store(true);
    }, true );

    const auto acmbPath = GetAcmbPath();

    MenuItemsHolder::GetInstance().AddItem( "Project", 2, "\xef\x83\x87", "Save", "Write the project to an .acmb file", [acmbPath] (size_t, bool)
    {
        FileDialog::Instance().OpenDialog( "SaveProjectDialog", "Save Table", ".acmb", ( acmbPath + "/GUI/presets/" ).c_str(), 1 );
    } );

    MenuItemsHolder::GetInstance().AddItem( "Project", 1, "\xef\x81\xbc", "Open", "Read the project from an .acmb file", [acmbPath] (size_t, bool)
    {
        FileDialog::Instance().OpenDialog( "OpenProjectDialog", "Load Table", ".acmb", ( acmbPath + "/GUI/presets/" ).c_str(), 1 );
    } );

    MenuItemsHolder::GetInstance().AddItem( "Help", 1, "\xef\x84\xa8", "Help", "Show modal window with instructions", [this] (size_t, bool)
    {
        _showHelpPopup = true;
    } );
}

void MainWindow::FinishProcessing()
{
    _isBusy = false;
    _showResultsPopup = true;
    UnlockInterface();
}

constexpr uint32_t U32Color( uint8_t r, uint8_t g, uint32_t b, uint32_t a )
{
    return r | ( g << 8 ) | ( b << 16 ) | ( a << 24 );
}
enum class UIColor : ImU32
{
    Arrow = U32Color( 0, 255, 0, 255 ),
    EmptyCell = U32Color( 32, 32, 32, 255 ),
    ActiveCellBorder = U32Color( 255, 0, 0, 255 ),
    TableBorders = U32Color( 64, 64, 64, 255 )
};

void MainWindow::ProcessKeyboardEvents()
{
    if ( ImGui::IsKeyPressed( ImGuiKey_LeftArrow ) )
        _selection.MoveLeft(GetPipelineSize(), _visibleCellsCount);
    if ( ImGui::IsKeyPressed( ImGuiKey_RightArrow ) )
        _selection.MoveRight(GetPipelineSize(), _visibleCellsCount);
    if ( _selection.elementSelected && ImGui::IsKeyPressed( ImGuiKey_Delete ) )
    {
        DeletePipelineElement(_pPipelineHead, size_t(_selection.activeIndex));
        UpdateActiveElement(_selection.activeIndex, _selection.elementSelected);
    }
}

void MainWindow::ProcessMouseEvents()
{
    if ( !ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) )
        return;

    const auto layout = GetPanelLayout();
    const float cGridBottom = layout.pipeline.GetBottom();
    const float cGridTop = layout.pipeline.y;
    if (layout.cellWidth <= 0)
        return;

    auto mousePos = ImGui::GetMousePos();
    mousePos.x -= _pos.x;
    mousePos.y -= _pos.y;
    if ( mousePos.y < cGridTop || mousePos.y > cGridBottom )
        return;
    if (mousePos.x < layout.pipeline.x || mousePos.x >= layout.pipeline.GetRight())
        return;
    const float gridX = mousePos.x - layout.pipeline.x;
    const int col = int(gridX / layout.cellWidth);
    bool isElementSelected = (gridX - col * layout.cellWidth) > cGridCellPadding * 2.0f;
    UpdateActiveElement(size_t(col) + _selection.firstVisible, isElementSelected);
}

void MainWindow::OpenProject()
{
    LoadProject( FileDialog::Instance().GetFilePathName() );
}

void MainWindow::RestoreAutoSavedProject()
{
    try
    {
        const auto filePath = std::filesystem::path( GetAcmbPath() ) / ".autosave.acmb";
        if ( std::filesystem::exists( filePath ) )
            LoadProject( filePath.string() );
    }
    catch ( const std::exception& e )
    {
        _errors.push_back( std::string( "Unable to restore pipeline autosave: " ) + e.what() );
        _showResultsPopup = true;
    }
}

bool MainWindow::LoadProject( const std::string& filePath )
{
    auto reportError = [this] ( const std::string msg )
    {
        ClearPipeline();
        _errors.push_back( msg );
        _showResultsPopup = true;
        return false;
    };

    try
    {
        ClearPipeline();

        std::ifstream fin( filePath, std::ios_base::in | std::ios_base::binary );

        _errors.clear();
        if ( !fin )
            return reportError( "Unable to open file" );
        fin.exceptions( std::ios_base::failbit | std::ios_base::badbit );

        fin.seekg( 0, std::ios_base::end );
        int streamSize = int( fin.tellg() );
        if ( streamSize < 5 )
            return reportError( "File is too small" );

        fin.seekg( 0 );

        std::string header( 4, '\0' );
        fin.read( &header[0], 4 );
        if ( header != "ACMB" )
            return reportError( "File is corrupted" );

        uint8_t pipelineSize = fin.get();

        if ( streamSize < pipelineSize + 5 )
            return reportError( "File is corrupted" );

        std::string serialized(pipelineSize, '\0' );
        fin.read( serialized.data(), pipelineSize);

        std::shared_ptr<PipelineElementWindow> pElement = nullptr;
        for ( size_t i = 0; i < pipelineSize; ++i )
        {
            const uint8_t menuOrder = static_cast<uint8_t>( serialized[i] );
            MenuItemsHolder::GetInstance().GetItems().at( "Tools" ).at( menuOrder )->action(i, true);
            if ( i == 0 )
                pElement = _pPipelineHead;
            else
                pElement = pElement->GetOutput();

            if ( !pElement || !pElement->Deserialize( fin ) )
                return reportError( "File is corrupted" );
        }

        if ( auto pReader = std::dynamic_pointer_cast<ImageReaderWindow>(_pPipelineHead) )
            pReader->OnFileListChanged();
        return true;
    }
    catch ( const std::exception& e )
    {
        return reportError( std::string( "Unable to load pipeline: " ) + e.what() );
    }
}

void MainWindow::SaveProject()
{
    _errors.clear();
    try
    {
        WriteProject( FileDialog::Instance().GetFilePathName() );
    }
    catch ( const std::exception& e )
    {
        _errors.push_back( e.what() );
        _showResultsPopup = true;
    }
}

void MainWindow::WriteProject( const std::string& filePath ) const
{
    std::ofstream fout;
    fout.exceptions( std::ios_base::failbit | std::ios_base::badbit );
    fout.open( filePath, std::ios_base::out | std::ios_base::binary );

    fout.write( "ACMB", 4 );

    const size_t pipelineSize = GetPipelineSize();    
    fout.put( char(pipelineSize) );

    std::string chars(pipelineSize, '\0' );

    auto pElement = _pPipelineHead;
    for ( int i = 0; i < pipelineSize; ++i )
    {
        chars[i] = pElement->GetMenuOrder();
        pElement = pElement->GetOutput();
    }        

    fout.write( chars.data(), chars.size() );
    
    pElement = _pPipelineHead;
    for ( int i = 0; i < pipelineSize; ++i )
    {
        pElement->Serialize( fout );
        pElement = pElement->GetOutput();
    }
    fout.close();
}

bool MainWindow::AutoSaveProject()
{
    try
    {
        const auto filePath = std::filesystem::path( GetAcmbPath() ) / ".autosave.acmb";
        const auto tempPath = std::filesystem::path( GetAcmbPath() ) / ".autosave.acmb.tmp";
#ifdef _WIN32
        // A previous failed replacement may have left the temporary file hidden.
        if ( std::filesystem::exists( tempPath ) && !SetFileAttributesW( tempPath.c_str(), FILE_ATTRIBUTE_NORMAL ) )
            throw std::runtime_error( "Unable to prepare pipeline autosave" );
#endif
        WriteProject( tempPath.string() );
#ifdef _WIN32
        if ( !SetFileAttributesW( tempPath.c_str(), FILE_ATTRIBUTE_HIDDEN ) ||
             !MoveFileExW( tempPath.c_str(), filePath.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH ) )
            throw std::runtime_error( "Unable to replace hidden pipeline autosave" );
#else
        std::filesystem::rename( tempPath, filePath );
#endif
        return true;
    }
    catch ( const std::exception& e )
    {
        _errors.push_back( std::string( "Unable to autosave pipeline: " ) + e.what() );
        _showResultsPopup = true;
        return false;
    }
}

bool MainWindow::PrepareForExit()
{
    _stopRequested.store( true );
    if ( _processingFuture.valid() )
        _processingFuture.wait();
    return AutoSaveProject();
}

PipelineElementWindow* MainWindow::GetActiveElement() const
{
    return GetPipelineElement(_pPipelineHead, size_t(_selection.activeIndex)).get();
}

void MainWindow::UpdateActiveElement(int newActiveElement, bool isElementSelected)
{
    _selection.Select(newActiveElement, isElementSelected, GetPipelineSize(), _visibleCellsCount);
}

void MainWindow::SetSize(const ImVec2& size)
{
    Window::SetSize(size);
    _visibleCellsCount = MainWindowLayout::CountVisibleCells(size.x);
    UpdateActiveElement(_selection.activeIndex, _selection.elementSelected);
}

void MainWindow::DrawMenu()
{    
    const auto layout = GetPanelLayout();
    if (layout.toolbox.width > 0 && layout.toolbox.height > 0)
    {
        ImGui::SetCursorPos({layout.toolbox.x, layout.toolbox.y});
        ImGui::BeginChild("##ToolboxSection", {layout.toolbox.width, layout.toolbox.height});

        ImVec2 cachedPos = ImGui::GetCursorPos();

        for ( const auto& it : MenuItemsHolder::GetInstance().GetItems() )
        {
            const std::string &category = it.first;
            const auto& items = it.second;

            ImGui::PushFont( _fontRegistry.bold );
            ImGui::SeparatorText( category.c_str() );
            ImGui::PopFont();

            const size_t itemCount = items.size();
            for ( size_t i = 0; i < itemCount; ++i)
            {
                const auto& item = *(std::next(items.begin(), i));
                if ( i > 0 && i % cItemCountInRow == 0 )
                {
                    ImGui::SetCursorPos( {0, cachedPos.y + cMenuRowHeight});
                }
                ImGui::PushFont( _fontRegistry.icons );

                cachedPos = ImGui::GetCursorPos();

                if ( item.second->unlockable )
                {
                    UI::UnlockableButton( item.second->icon, { cMenuButtonSize, cMenuButtonSize }, [&]
                    {
                        item.second->action(_selection.activeIndex, _selection.elementSelected);
                    }, item.second->tooltip );
                }
                else
                {
                    const bool isAnyPopupOpen = ImGui::IsPopupOpen( "", ImGuiPopupFlags_AnyPopupId );
                    UI::Button( item.second->icon, { cMenuButtonSize, cMenuButtonSize }, [&]
                    {
                        if ( !isAnyPopupOpen )
                            item.second->action(_selection.activeIndex, _selection.elementSelected);
                    }, isAnyPopupOpen ? "Menu is disabled while any ancillary window is opened" : item.second->tooltip );
                }

                ImGui::PopFont();

                const float textWidth = ImGui::CalcTextSize( item.second->caption.c_str() ).x;
                ImGui::SetCursorPos( { cachedPos.x + ( cMenuButtonSize - textWidth ) * 0.5f, cachedPos.y + cMenuButtonSize + ImGui::GetStyle().ItemSpacing.y } );
                ImGui::Text( "%s", item.second->caption.c_str() );
                ImGui::SetCursorPos( { cachedPos.x + cMenuButtonSize + ImGui::GetStyle().ItemSpacing.x, cachedPos.y});

                cachedPos = ImGui::GetCursorPos();
            }

            ImGui::SetCursorPos({ 0, cachedPos.y + cMenuRowHeight + 3.0f * ImGui::GetStyle().ItemSpacing.y });
            cachedPos = ImGui::GetCursorPos();
        }



        if ( _fontRegistry.bold )
            ImGui::PushFont( _fontRegistry.bold );

        ImGui::SeparatorText( "General Settings" );

        if ( _fontRegistry.bold )
            ImGui::PopFont();

        ImGui::SetCursorPos( { cachedPos.x, cachedPos.y + ImGui::GetTextLineHeight() + ImGui::GetStyle().ItemSpacing.y } );
        if ( cuda::isCudaAvailable() )
            UI::Checkbox( "Enable CUDA", &_enableCuda, "Performs computations on a graphic card if available" );

        ImGui::EndChild();
    }

    auto fileDialog = FileDialog::Instance();
    if ( fileDialog.Display( "OpenProjectDialog", {}, { 300 * cMenuScaling, 200 * cMenuScaling } ) )
    {
        if ( fileDialog.IsOk() )
            OpenProject();

        fileDialog.Close();
    }

    if ( fileDialog.Display( "SaveProjectDialog", {}, { 300 * cMenuScaling, 200 * cMenuScaling } ) )
    {
        if ( fileDialog.IsOk() )
            SaveProject();

        fileDialog.Close();
    }
}

void MainWindow::UpdateProcessing()
{
    if ( _processingFuture.valid() &&
         _processingFuture.wait_for(std::chrono::seconds(0)) == std::future_status::ready )
    {
        try
        {
            // get() rethrows exceptions captured at the async worker boundary.
            _errors = _processingFuture.get();
        }
        catch ( const std::exception& e )
        {
            _errors = { e.what() };
        }
        catch ( ... )
        {
            _errors = { "Unknown error while processing images" };
        }
        FinishProcessing();
    }
}

void MainWindow::DrawDialog()
{
    UpdateProcessing();
    if ( IsInterfaceLocked() && !ImGui::IsPopupOpen( "", ImGuiPopupFlags_AnyPopupId ) && !FileDialog::Instance().IsOpened() && !_isBusy )
        UnlockInterface();

    if ( !_lockInterface && !_showHelpPopup )
    {
        ProcessKeyboardEvents();
        ProcessMouseEvents();
    }

    DrawMenu();

    ImGui::PushFont( _fontRegistry.byDefault );

    const auto layout = GetPanelLayout();
    _visibleCellsCount = layout.visibleCells;
    const float cGridBottom = layout.pipeline.GetBottom();
    const float cGridTop = layout.pipeline.y;
    const float cBoldLineThickness = 3.0f;

    auto drawList = ImGui::GetWindowDrawList();
    drawList->AddLine({layout.toolboxSeparatorX, 0}, {layout.toolboxSeparatorX, layout.pipelineHeader.y}, ImGui::GetColorU32(ImGui::GetStyleColorVec4(ImGuiCol_Separator)), 3.0f);
    drawList->AddLine({layout.inspectorSeparatorX, 0}, {layout.inspectorSeparatorX, layout.pipelineHeader.y}, ImGui::GetColorU32(ImGui::GetStyleColorVec4(ImGuiCol_Separator)), 3.0f);

    auto pElement = GetActiveElement();    
    const auto imageRegionAvail = GetImageRegionAvail();   

    if (layout.frameControls.width > 0 && layout.frameControls.height > 0)
    {
        ImGui::SetCursorPos({layout.frameControls.x, layout.frameControls.y});
        ImGui::BeginChild("##FrameCounterSection", {layout.frameControls.width, layout.frameControls.height});

        ImGui::PushFont(_fontRegistry.bold);
        ImGui::SeparatorText("Frame Number Selection");
        ImGui::PopFont();

        if ( pElement && pElement->GetTaskCount() > 1 )
        {
            const float buttonWidth = 150.0f;
            const float spacing = ImGui::GetStyle().ItemSpacing.x;
            const float totalWidth = buttonWidth * 2 + spacing;

            int frameNumber = pElement->GetPreviewedFrameNumber();
            ImGui::SetNextItemWidth(totalWidth - ImGui::CalcTextSize("Frame Number").x - spacing);
            if ( UI::SliderInt("Frame Number", &frameNumber, 0, int(pElement->GetTaskCount()) - 1, "Select frame number", pElement) )
            {
                pElement->OnPreviewedFrameNumberChanged(frameNumber);
            }

            if ( auto pSettingsInterpolationUser = dynamic_cast<ISettingsInterpolationUser*>(pElement) )
            {
                pSettingsInterpolationUser->DrawFrameCounter();
            }
        }

        ImGui::EndChild();
    }

    if ( pElement && imageRegionAvail.width > 0 && imageRegionAvail.height > 0 )
    {
        if ( auto textureOpt = pElement->GetPreviewTexture(); textureOpt.has_value() && textureOpt.value() && textureOpt.value()->GetTexture() != nullptr)
        {
            auto pTexture = textureOpt.value();

            const uint32_t width = pTexture->GetWidth();
            const uint32_t height = pTexture->GetHeight();

            const float aspectRatio = float(width) / float(height);
            const float imageRegionAspect = float(imageRegionAvail.width) / float(imageRegionAvail.height);

            ImVec2 topLeftPos;
            ImVec2 scaleFactors;

            if ( width < imageRegionAvail.width && height < imageRegionAvail.height )
            {
                topLeftPos.x = float(imageRegionAvail.x + (imageRegionAvail.width - width) * 0.5f);
                topLeftPos.y = float(imageRegionAvail.y + (imageRegionAvail.height - height) * 0.5f);
            }
            else if ( aspectRatio > imageRegionAspect )
            {
                topLeftPos.x = float(imageRegionAvail.x);
                topLeftPos.y = float(imageRegionAvail.y + (imageRegionAvail.height - (imageRegionAvail.width / aspectRatio)) * 0.5f);

            }
            else
            {
                topLeftPos.x = float(imageRegionAvail.x + (imageRegionAvail.width - (imageRegionAvail.height * aspectRatio)) * 0.5f);
                topLeftPos.y = float(imageRegionAvail.y);
            }

            scaleFactors.x = float(width) / float(width);

            ImGui::SetCursorPos(topLeftPos);
            ImGui::Image(pTexture->GetTexture(), { float(width), float(height) });
            pElement->DrawOnPreviewImage(drawList, topLeftPos, { float(width), float(height) });
        }
    }    

    drawList->AddLine({ 0, cGridTop - cBoldLineThickness }, { _size.x - 6, cGridTop - cBoldLineThickness }, ImGui::GetColorU32(ImGui::GetStyleColorVec4(ImGuiCol_Separator)), 3.0f);
    drawList->AddLine({ 0, cGridTop - cHeadRowHeight - cBoldLineThickness }, { _size.x - 6, cGridTop - cHeadRowHeight - cBoldLineThickness }, ImGui::GetColorU32(ImGui::GetStyleColorVec4(ImGuiCol_Separator)), 3.0f);
    drawList->AddLine({ 0, cGridBottom }, { _size.x - 6, cGridBottom }, ImGui::GetColorU32(ImGui::GetStyleColorVec4(ImGuiCol_Separator)), 3.0f);

    drawList->AddLine({ 5, cGridTop - cHeadRowHeight - 1 }, { 5, cGridBottom }, ImU32(UIColor::TableBorders), 3.0f);


    ImGui::SetCursorPos({5, cGridTop - cHeadRowHeight - cBoldLineThickness * 0.5f });
    UI::Button( "##ClearPipeline", { cGridLeft - 5, cHeadRowHeight - cBoldLineThickness * 0.5f }, [this]
    {
        ClearPipeline();
    }, "Clear pipeline" );

    ImVec2 topLeft;
    ImVec2 bottomRight;

    _gridCellSize.width = layout.cellWidth;

    for ( int x = 0; x < int(_visibleCellsCount); ++x )
    {
        topLeft.x = float( cGridLeft + x * _gridCellSize.width);
        drawList->AddLine( { topLeft.x - 1, cGridTop - cHeadRowHeight - 1 }, { topLeft.x - 1, cGridBottom }, ImU32( UIColor::TableBorders ) );
        ImGui::SetCursorPos( { topLeft.x + _gridCellSize.width * 0.5f, cGridTop - cHeadRowHeight + ImGui::GetTextLineHeightWithSpacing() * 0.25f } );

        std::string columnHeader = std::to_string( x + _selection.firstVisible + 1 );
        ImGui::Text( "%s", columnHeader.c_str() );
    }

    topLeft.x = float(cGridLeft + _visibleCellsCount * _gridCellSize.width);
    drawList->AddLine( { _size.x - 8, cGridTop - cHeadRowHeight - 1 }, { _size.x - 8, cGridBottom }, ImU32( UIColor::TableBorders ), 3.0f );

    
    topLeft.y = cGridTop;
    bottomRight.y = topLeft.y + cGridCellHeight;
    
    ImGui::SetCursorPos( { cGridLeft * 0.5f, topLeft.y + cGridCellHeight * 0.5f - ImGui::GetTextLineHeightWithSpacing() * 0.5f } );    
    ImGui::Text("%s", "1");

    const float gridCellPaddingX = ( _gridCellSize.width - PipelineElementWindow::cElementWidth ) * 0.5f;

    pElement = _pPipelineHead.get();
    for ( int i = 0; i < _selection.firstVisible; ++i )
    {
        pElement = pElement->GetOutput().get();
    }

    for ( int x = 0; x < int(_visibleCellsCount); ++x )
    {
        topLeft.x = float( cGridLeft + x * _gridCellSize.width);
        bottomRight.x = topLeft.x + _gridCellSize.width;

        const size_t elementIdx = x + _selection.firstVisible;

        ImGui::PushClipRect( { topLeft.x - 1, topLeft.y - 1 }, bottomRight, false );

        if ( _selection.activeIndex == elementIdx )
        {
            if ( _selection.elementSelected )
                drawList->AddRect({ topLeft.x + gridCellPaddingX - 1, topLeft.y + cGridCellPadding - 1 }, { bottomRight.x - gridCellPaddingX + 1, bottomRight.y - cGridCellPadding + 1 }, ImU32(UIColor::ActiveCellBorder), 0, 0, 2.0f);
            else
                drawList->AddLine( { topLeft.x, topLeft.y}, { topLeft.x, bottomRight.y }, ImU32( UIColor::ActiveCellBorder ), 2.0f );
        }

        if ( pElement )
        {
            pElement->SetPos( { topLeft.x + gridCellPaddingX, topLeft.y + cGridCellPadding } );
            pElement = pElement->GetOutput().get();
            ImGui::PopClipRect();
            continue;
        }
        
        drawList->AddRectFilled( { topLeft.x + gridCellPaddingX - 1, topLeft.y + cGridCellPadding - 1 }, { bottomRight.x - gridCellPaddingX + 1, bottomRight.y - cGridCellPadding + 1 }, ImU32( UIColor::EmptyCell ) );           
        
        if ( _selection.activeIndex == elementIdx )
        {
            if ( _selection.elementSelected )
                drawList->AddRect({ topLeft.x + gridCellPaddingX - 1, topLeft.y + cGridCellPadding - 1 }, { bottomRight.x - gridCellPaddingX + 1, bottomRight.y - cGridCellPadding + 1 }, ImU32(UIColor::ActiveCellBorder), 0, 0, 2.0f);
            else
                drawList->AddLine({ topLeft.x, topLeft.y }, { topLeft.x, bottomRight.y }, ImU32(UIColor::ActiveCellBorder), 2.0f);
        }

        ImGui::PopClipRect();
    }

    //topLeft.y = float( cGridTop + _viewportSize.height * cGridCellHeight );
    //drawList->AddLine( { 0, topLeft.y - 1 }, { _size.x, topLeft.y - 1 }, ImU32( UIColor::TableBorders ) );

    ImGui::PopFont();

    if ( _showResultsPopup )
    {
        if ( _durationString.empty() )
        {
            const auto ms = std::chrono::duration_cast< std::chrono::milliseconds >(std::chrono::high_resolution_clock::now() - _startTime).count();
            _durationString = "Elapsed " + std::to_string( ms / 1000 ) + "s " + std::to_string( ms % 1000 ) + "ms";
            std::cout << _durationString << std::endl;
        }

        if ( _errors.empty() )
        {
            if ( ImGui::IsPopupOpen( "", ImGuiPopupFlags_AnyPopupId ) && !ImGui::IsPopupOpen( " Success##modal" ) )
            {
                return;
            }

            return UI::ShowModalMessage( { _durationString }, UI::ModalMessageType::Success, _showResultsPopup );
        }

        return UI::ShowModalMessage( _errors, UI::ModalMessageType::Error, _showResultsPopup );
    }

    if ( _showHelpPopup )
    {
        UI::ShowModalMessage( 
            { "1. Each cell may contain a tool that imports, processes, or exports images to disk.\n"
              "2. The schema must contain at least one 'Import' and 'Export' tool.\n"
              "3. Tools (except for 'Import') accept images as input from the adjacent cell to the left or top of themselves.\n"
              "4. Tools (except for 'Import') (except 'Export') transmit read/processed images to the adjacent right or bottom cell\n"
              "5. Images between tools connected by three parallel arrows are transmitted one by one, in batch mode.\n"
              "6. Images between instruments connected by converging lines are first summed up, and then processed by the receiving instrument.\n"
              "7. The type of connection between the tools can be changed by double-clicking \n"
              "8. To learn more, follow the link https://github.com/astrowander/acmb#readme" },
            UI::ModalMessageType::Help, _showHelpPopup );
    }
}

#ifdef _WIN32
static CRect GetWorkingArea()
{
    CRect rcDesktop;
    ::SystemParametersInfo( SPI_GETWORKAREA, NULL, &rcDesktop, NULL );
    return rcDesktop;
}
#endif

static std::pair<ImVec2, ImVec2> GetWindowRect()
{
#ifdef _WIN32
    const auto rcDesktop = GetWorkingArea();
    return {{float( rcDesktop.left ), float( rcDesktop.top )}, {float( rcDesktop.Width() ), float( rcDesktop.Height() )} };
#elif defined ( __linux__ )
    const auto pVideoMode = glfwGetVideoMode( glfwGetPrimaryMonitor() );
    return { { 0.0f, 0.0f}, {float( pVideoMode->width ), float( pVideoMode->height ) } };
#endif
}

MainWindow& MainWindow::GetInstance( const FontRegistry& fontRegistry )
{
    static const auto windowRect = GetWindowRect();
    static auto pInstance = std::unique_ptr<MainWindow>( new MainWindow( windowRect.first, windowRect.second, fontRegistry));
    return *pInstance;
}

RectF MainWindow::GetImageRegionAvail() const
{
    return GetPanelLayout().preview;
}

MainWindowLayout MainWindow::GetPanelLayout() const
{
    const auto& style = ImGui::GetStyle();
    return MainWindowLayout::Calculate(_size, style.ItemSpacing,
        style.WindowPadding, ImGui::GetTextLineHeightWithSpacing());
}

void MainWindow::Show()
{
    if ( !DrawHeader() )
        return ImGui::End();

    const int pipelineSize = int( GetPipelineSize() );
    auto pElement = _pPipelineHead;

    for ( int i = 0; i < _selection.firstVisible; ++i )
    {
        pElement = pElement->GetOutput();
    }

    for ( int i = 0; i < std::min(_visibleCellsCount, pipelineSize - _selection.firstVisible); ++i )
    {
        pElement->Show();
        pElement = pElement->GetOutput();
    }

    ImGui::SetNextWindowPos(_pos, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(_size, ImGuiCond_FirstUseEver);

    DrawDialog();
    ImGui::End();   
}

ACMB_GUI_NAMESPACE_END
