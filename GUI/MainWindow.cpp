#include "MainWindow.h"
#include "ImageReaderWindow.h"
#include "ImageWriterWindow.h"
#include "SettingsInterpolationUser.h"
#include "FontRegistry.h"
#include "FileDialog.h"
#include "ImGuiHelpers.h"
#include "TopToolbar.h"
#include "ToolLibrary.h"
#include "AdaptivePanels.h"
#include "StepPresentation.h"
#include <cstdlib>
#include "./../Tools/SystemTools.h"
#include "./../Cuda/CudaInfo.h"
#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <sstream>

#ifdef _WIN32
#include <atltypes.h>
#include <WinUser.h>
#elif defined ( __linux__ )
#include <GLFW/glfw3.h>
#endif

ACMB_GUI_NAMESPACE_BEGIN

static void DrawPanelCaption(const char* caption, ImFont* font, float width = 0)
{
    const float left = ImGui::GetCursorPosX();
    if (width <= 0) width = ImGui::GetContentRegionAvail().x;
    ImGui::PushFont(font);
    ImGui::SetCursorPosX(left + std::max(0.0f, (width - ImGui::CalcTextSize(caption).x) * 0.5f));
    ImGui::TextUnformatted(caption);
    ImGui::PopFont();
    ImGui::SetCursorPosX(left);
    const auto line = ImGui::GetCursorScreenPos();
    ImGui::GetWindowDrawList()->AddLine(line, {line.x + width, line.y},
        ImGui::GetColorU32(ImGuiCol_Separator));
    ImGui::Dummy({width, 1.0f});
    ImGui::SetCursorPosX(left);
}

static std::filesystem::path UserLayoutPreferencesPath()
{
#ifdef _WIN32
#ifdef _MSC_VER
    char* local = nullptr;
    size_t length = 0;
    _dupenv_s(&local, &length, "LOCALAPPDATA");
    const std::unique_ptr<char, decltype(&std::free)> ownedLocal(local, &std::free);
    if (local && *local)
#else
    if (const auto local = std::getenv("LOCALAPPDATA"); local && *local)
#endif
        return std::filesystem::path(local) / "AstroCombine" / "layout.ini";
#else
    if (const auto config = std::getenv("XDG_CONFIG_HOME"); config && *config)
        return std::filesystem::path(config) / "AstroCombine" / "layout.ini";
    if (const auto homeDirectory = std::getenv("HOME"); homeDirectory && *homeDirectory)
        return std::filesystem::path(homeDirectory) / ".config" / "AstroCombine" / "layout.ini";
#endif
    return std::filesystem::path(GetAcmbPath()) / "GUI" / "layout.ini";
}

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
    _layoutPreferencesPath = UserLayoutPreferencesPath();
    _layoutPreferences.Load(_layoutPreferencesPath);
    SetPos( pos );
    _visibleCellsCount = GetPanelLayout().visibleCells;

    MenuItemsHolder::GetInstance().AddItem( "Run", 1, "\xef\x81\x8B", "Run", "Start processing", [this] (size_t, bool)
    {
        if ( _processingFuture.valid() )
            return;

        _errors.clear();
        if ( !AutoSaveProject() )
            return;
        _showResultsPopup = false;
        _processingState.Begin();
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
        _processingState.RequestStop();
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
    _processingState.Finish();
    _showResultsPopup = true;
    UnlockInterface();
}

void MainWindow::ProcessKeyboardEvents()
{
    if (ImGui::GetIO().WantTextInput || ImGui::IsAnyItemActive())
        return;
    ProcessSelectionKeys(_selection, GetPipelineSize(), _visibleCellsCount);
    if ( _selection.elementSelected && ImGui::IsKeyPressed( ImGuiKey_Delete ) )
    {
        DeletePipelineElement(_pPipelineHead, size_t(_selection.activeIndex));
        UpdateActiveElement(_selection.activeIndex, _selection.elementSelected);
    }
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
    if (_layoutPreferencesDirty) SaveLayoutPreferences();
    _processingState.RequestShutdown();
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
    _visibleCellsCount = GetPanelLayout().visibleCells;
    UpdateActiveElement(_selection.activeIndex, _selection.elementSelected);
}

void MainWindow::DrawToolbar()
{
    const auto layout = GetPanelLayout();
    if (layout.toolbar.width <= 0 || layout.toolbar.height <= 0)
        return;
    const auto& items = MenuItemsHolder::GetInstance().GetItems();
    const ToolbarActions actions{items.at("Project").at(1).get(), items.at("Project").at(2).get(),
        items.at("Run").at(1).get(), items.at("Run").at(2).get(), items.at("Help").at(1).get()};
    ImGui::SetCursorPos({layout.toolbar.x, layout.toolbar.y});
    ImGui::PushFont(_fontRegistry.byDefault);
    DrawTopToolbar({layout.toolbar.width, layout.toolbar.height}, actions,
        [this] { return IsInterfaceLocked() || _showHelpPopup || _showResultsPopup || FileDialog::Instance().IsOpened(); },
        size_t(_selection.activeIndex), _selection.elementSelected, cuda::isCudaAvailable(), _enableCuda,
        {}, [this] { DrawLayoutMenu(); });
    ImGui::PopFont();
    ImGui::SetCursorPos({layout.toolbar.x, layout.toolbar.GetBottom()});
    ImGui::Separator();
}

void MainWindow::SaveLayoutPreferences()
{
    _layoutPreferences.Clamp();
    _layoutPreferencesSaveFailed = !_layoutPreferences.Save(_layoutPreferencesPath);
    _layoutPreferencesDirty = _layoutPreferencesSaveFailed;
}

void MainWindow::ResetLayout()
{
    _layoutPreferences = {};
    _lastPipelineSelection = -1;
    _lastPipelineWidth = 0;
    SaveLayoutPreferences();
}

void MainWindow::DrawLayoutMenu()
{
    if (!ImGui::BeginMenu("View")) return;
    const auto layout = GetPanelLayout();
    if (ImGui::MenuItem("Tools drawer", nullptr, false, layout.mode != PanelMode::Wide))
        _toolsDrawerRequested = true;
    if (ImGui::MenuItem("Inspector drawer", nullptr, false, layout.mode == PanelMode::Narrow))
        _inspectorDrawerRequested = true;
    if (ImGui::MenuItem("Collapse pipeline", nullptr, &_layoutPreferences.pipelineCollapsed))
    {
        _lastPipelineSelection = -1;
        SaveLayoutPreferences();
    }
    ImGui::Separator();
    if (ImGui::MenuItem("Reset layout")) ResetLayout();
    if (_layoutPreferencesSaveFailed)
    {
        ImGui::TextDisabled("Layout preferences not saved");
        ImGui::SetItemTooltip("Could not write %s", _layoutPreferencesPath.string().c_str());
    }
    ImGui::EndMenu();
}

void MainWindow::DrawToolsContent(bool closeOnAction)
{
    DrawPanelCaption("Toolbox", _fontRegistry.bold);
    DrawToolLibrary(MenuItemsHolder::GetInstance().GetToolGroups(),
        [this] { return IsInterfaceLocked(); }, size_t(_selection.activeIndex), _selection.elementSelected,
        [this](const MenuItem& item) { UI::SetTooltipIfHovered(item.tooltip, cMenuScaling); }, {}, closeOnAction);
}

void MainWindow::DrawInspectorContent()
{
    DrawPanelCaption("Tool Settings", _fontRegistry.bold);
    auto element = GetActiveElement();
    if (!element || !_selection.elementSelected)
    {
        ImGui::TextWrapped("Select a pipeline step to edit its settings.");
        return;
    }
    const float availableHeight = ImGui::GetContentRegionAvail().y;
    const float baseFooterHeight = 2 * ImGui::GetTextLineHeightWithSpacing() + ImGui::GetFrameHeight() +
        3 * ImGui::GetStyle().ItemSpacing.y + 2 * ImGui::GetStyle().WindowPadding.y + 1;
    const float footerHeight = _histogramExpanded
        ? std::max(baseFooterHeight, std::min(baseFooterHeight + 150 * GetPanelLayout().scale, availableHeight * 0.5f))
        : baseFooterHeight;
    const float settingsHeight = std::max(1.0f, availableHeight - footerHeight - ImGui::GetStyle().ItemSpacing.y);
    if (ImGui::BeginChild("##InspectorSettings", {0, settingsHeight}, ImGuiChildFlags_None,
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse))
        element->DrawSettings();
    ImGui::EndChild();
    if (ImGui::BeginChild("##InspectorFooter", {0, footerHeight}))
    {
        ImGui::Separator();
        if (auto params = element->GetImageParams())
        {
            ImGui::Text("Size: %u x %u", params->GetWidth(), params->GetHeight());
            std::ostringstream format;
            format << params->GetPixelFormat();
            ImGui::Text("Pixel format: %s", format.str().c_str());
        }
        else
        {
            ImGui::TextDisabled("Size: unavailable");
            ImGui::TextDisabled("Pixel format: unavailable");
        }
        ImGui::SetNextItemOpen(_histogramExpanded, ImGuiCond_Always);
        _histogramExpanded = ImGui::CollapsingHeader("Preview Histogram");
        if (_histogramExpanded)
            DrawPreviewHistogram(element);
    }
    ImGui::EndChild();
}

void MainWindow::DrawPreviewHistogram(PipelineElementWindow* element)
{
    ImGui::Checkbox("Logarithmic scale", &_histogramLogarithmic);
    auto preview = element->GetPreviewData();
    const bool updating = !preview;
    if (updating)
        preview = element->GetDisplayedPreviewData();
    if (!preview)
    {
        ImGui::TextDisabled("Preview not available yet.");
        return;
    }
    const ImU32 colors[] = {IM_COL32(255, 90, 90, 255), IM_COL32(90, 230, 110, 255), IM_COL32(100, 150, 255, 255)};
    float maximum = 1;
    for (const auto& channel : preview->histogram)
        for (float count : channel) maximum = std::max(maximum, count);
    const float scaleMaximum = _histogramLogarithmic ? std::log1p(maximum) : maximum;
    const float scale = GetPanelLayout().scale;
    const ImVec2 size{std::max(1.0f, ImGui::GetContentRegionAvail().x),
        std::clamp(ImGui::GetContentRegionAvail().y - ImGui::GetTextLineHeightWithSpacing(), 40 * scale, 120 * scale)};
    ImGui::InvisibleButton("##PreviewHistogramPlot", size);
    const auto topLeft = ImGui::GetItemRectMin();
    const auto bottomRight = ImGui::GetItemRectMax();
    auto* draw = ImGui::GetWindowDrawList();
    draw->AddRectFilled(topLeft, bottomRight, ImGui::GetColorU32(ImGuiCol_FrameBg));
    const ImVec2 plotMin{topLeft.x + 1, topLeft.y + 1};
    const ImVec2 plotMax{std::max(plotMin.x, bottomRight.x - 1), std::max(plotMin.y, bottomRight.y - 1)};
    // Polyline joins and their antialiasing can extend beyond the input points.
    draw->PushClipRect(plotMin, plotMax, true);
    const bool grayscale = preview->grayscale;
    for (int channel = 0; channel < (grayscale ? 1 : 3); ++channel)
    {
        std::array<ImVec2, 256> points;
        for (int bin = 0; bin < 256; ++bin)
        {
            const float count = preview->histogram[channel][bin];
            const float height = std::clamp(
                (_histogramLogarithmic ? std::log1p(count) : count) / scaleMaximum, 0.0f, 1.0f);
            points[bin] = {plotMin.x + (plotMax.x - plotMin.x) * bin / 255.0f,
                plotMax.y - (plotMax.y - plotMin.y) * height};
        }
        draw->AddPolyline(points.data(), int(points.size()),
            grayscale ? ImGui::GetColorU32(ImGuiCol_Text) : colors[channel], ImDrawFlags_None, 1.0f);
    }
    draw->PopClipRect();
    if (updating)
        draw->AddRectFilled(topLeft, bottomRight, IM_COL32(0, 0, 0, 120));
    draw->AddRect(topLeft, bottomRight, ImGui::GetColorU32(ImGuiCol_Border));
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Preview brightness: 0 (left) to 255 (right).\nHeight: %s, shared across channels.",
            _histogramLogarithmic ? "log(1 + pixel count)" : "pixel count (linear)");
    ImGui::TextDisabled(updating ? "Updating preview..." :
        (grayscale ? "Grayscale | 0 - 255" : "RGB channels | 0 - 255"));
}

void MainWindow::DrawAdaptivePanels(const MainWindowLayout& layout)
{
    if (layout.drawerBar.width > 0)
    {
        ImGui::SetCursorPos({layout.drawerBar.x, layout.drawerBar.y});
        if (ImGui::Button("Tools")) _toolsDrawerRequested = true;
        if (layout.mode == PanelMode::Narrow)
        {
            ImGui::SameLine();
            if (ImGui::Button("Inspector")) _inspectorDrawerRequested = true;
        }
    }
    const auto origin = ImGui::GetWindowPos();
    if (BeginLayoutDrawer("Tools drawer", "Tools", layout.toolsDrawer, origin, _toolsDrawerRequested))
    {
        if (BeginVerticalPanel("toolDrawerContent", {0, 0}, 220)) DrawToolsContent(true);
        ImGui::EndChild();
        ImGui::EndPopup();
    }
    if (BeginLayoutDrawer("Inspector drawer", "Inspector", layout.inspectorDrawer, origin, _inspectorDrawerRequested))
    {
        DrawInspectorContent();
        ImGui::EndPopup();
    }
}

void MainWindow::DrawSplitters(const MainWindowLayout& layout)
{
    const auto toolbox = DrawLayoutSplitter("toolboxSplitter", layout.toolboxSplitter, true);
    _layoutPreferencesDirty |= ResizeLayoutPanel(_layoutPreferences, layout, LayoutSplitter::Toolbox, toolbox.delta);
    const auto inspector = DrawLayoutSplitter("inspectorSplitter", layout.inspectorSplitter, true);
    _layoutPreferencesDirty |= ResizeLayoutPanel(_layoutPreferences, layout, LayoutSplitter::Inspector, inspector.delta);
    SplitterDrag pipeline;
    if (!layout.pipelineCollapsed)
        pipeline = DrawLayoutSplitter("pipelineSplitter", layout.pipelineSplitter, false);
    else if (layout.pipelineSplitter.width > 0)
    {
        ImGui::SetCursorPos({layout.pipelineSplitter.x,
            layout.pipelineSplitter.y + layout.pipelineSplitter.height * 0.5f});
        ImGui::Separator();
    }
    _layoutPreferencesDirty |= ResizeLayoutPanel(_layoutPreferences, layout, LayoutSplitter::Pipeline, pipeline.delta);
    if (_layoutPreferencesDirty && (toolbox.released || inspector.released || pipeline.released))
        SaveLayoutPreferences();
}

void MainWindow::DrawMenu()
{    
    const auto layout = GetPanelLayout();
    if (layout.toolbox.width > 0 && layout.toolbox.height > 0)
    {
        ImGui::SetCursorPos({layout.toolbox.x, layout.toolbox.y});
        ImGui::PushFont(_fontRegistry.byDefault);
        if (BeginVerticalPanel("##ToolboxSection", {layout.toolbox.width, layout.toolbox.height}, 220))
        {
            DrawToolsContent();
        }
        ImGui::EndChild();
        ImGui::PopFont();
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

void MainWindow::DrawKeyframes(const MainWindowLayout& layout)
{
    ImGui::SetCursorPos({layout.keyframes.x, layout.keyframes.y});
    if (ImGui::BeginChild("##KeyframesPanel", {layout.keyframes.width, layout.keyframes.height},
        ImGuiChildFlags_Borders))
    {
        const auto& panelStyle = ImGui::GetStyle();
        const auto controlsOrigin = ImGui::GetCursorPos();
        const float sliderX = layout.pipeline.x - layout.keyframes.x;
        const float buttonsWidth = layout.pipelineCommands.width - 2 * panelStyle.WindowPadding.x;
        auto element = _selection.elementSelected ? GetActiveElement() : nullptr;
        auto interpolation = dynamic_cast<ISettingsInterpolationUser*>(element);
        const auto panelOrigin = ImGui::GetWindowPos();
        const float dividerX = panelOrigin.x + sliderX - panelStyle.ItemSpacing.x * 0.5f;
        ImGui::GetWindowDrawList()->AddLine(
            {dividerX, panelOrigin.y + panelStyle.WindowPadding.y},
            {dividerX, panelOrigin.y + ImGui::GetWindowHeight() - panelStyle.WindowPadding.y},
            ImGui::GetColorU32(ImGuiCol_Separator));
        DrawPanelCaption("Manage Keyframes", _fontRegistry.bold, buttonsWidth);
        if (interpolation)
        {
            ImGui::PushID(element);
            ImGui::BeginDisabled(IsInterfaceLocked() || element->GetTaskCount() <= 1);
            UI::SettingsParentScope parent(element);
            interpolation->DrawFrameCounter(buttonsWidth);
            ImGui::EndDisabled();
            ImGui::PopID();
        }
        else
        {
            const char* message = element ? "No keyframes for this tool" : "No tool selected";
            const auto textSize = ImGui::CalcTextSize(message, nullptr, false, buttonsWidth);
            const float top = ImGui::GetCursorPosY();
            const float availableHeight = std::max(0.0f,
                ImGui::GetWindowHeight() - panelStyle.WindowPadding.y - top);
            ImGui::SetCursorPos({controlsOrigin.x + std::max(0.0f, (buttonsWidth - textSize.x) * 0.5f),
                top + std::max(0.0f, (availableHeight - textSize.y) * 0.5f)});
            ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + textSize.x);
            ImGui::TextUnformatted(message);
            ImGui::PopTextWrapPos();
        }
        if (element && element->GetTaskCount() > 1)
        {
            ImGui::PushID(element);
            ImGui::BeginDisabled(IsInterfaceLocked());
            UI::SettingsParentScope parent(element);
            ImGui::SetCursorPos({sliderX, controlsOrigin.y});
            const float sliderWidth = std::max(1.0f, ImGui::GetContentRegionAvail().x);
            ImGui::SetCursorPosX(sliderX + std::max(0.0f,
                (sliderWidth - ImGui::CalcTextSize("Select frame number").x) * 0.5f));
            ImGui::TextUnformatted("Select frame number");
            ImGui::SetCursorPosX(sliderX);
            ImGui::PushItemWidth(sliderWidth);
            int frame = element->GetPreviewedFrameNumber();
            if (UI::SliderInt("##FrameNumber", &frame, 0, int(element->GetTaskCount()) - 1,
                "Select frame number", element))
                element->OnPreviewedFrameNumberChanged(frame);
            if (auto interpolation = dynamic_cast<ISettingsInterpolationUser*>(element))
            {
                const auto sliderMin = ImGui::GetItemRectMin();
                const auto sliderMax = ImGui::GetItemRectMax();
                const int lastFrame = int(element->GetTaskCount()) - 1;
                // Match ImGui's integer slider grab centers, including its end padding.
                const float trackWidth = std::max(0.0f, sliderMax.x - sliderMin.x - 4.0f);
                const float grabWidth = std::min(trackWidth,
                    std::max(panelStyle.GrabMinSize, trackWidth / float(lastFrame + 1)));
                const float firstX = sliderMin.x + 2.0f + grabWidth * 0.5f;
                const float travel = trackWidth - grabWidth;
                auto* drawList = ImGui::GetWindowDrawList();
                const auto tickColor = ImGui::GetColorU32(ImGuiCol_Text);
                for (int keyframe : interpolation->GetKeyframeNumbers())
                {
                    if (keyframe < 0 || keyframe > lastFrame) continue;
                    const float x = firstX + travel * float(keyframe) / float(lastFrame);
                    drawList->AddLine({x, sliderMax.y + layout.scale},
                        {x, sliderMax.y + 6.0f * layout.scale}, tickColor, layout.scale);
                }
            }
            ImGui::PopItemWidth();
            ImGui::EndDisabled();
            ImGui::PopID();
        }
        else
        {
            ImGui::SetCursorPos({sliderX, controlsOrigin.y});
            ImGui::TextDisabled("Select a pipeline step with multiple frames.");
        }
    }
    ImGui::EndChild();
}

void MainWindow::DrawPipeline(const MainWindowLayout& layout)
{
    if (layout.pipeline.width <= 0 || layout.pipeline.height <= 0)
        return;
    const auto& panelStyle = ImGui::GetStyle();
    ImGui::SetCursorPos({layout.pipelineCommands.x, layout.pipelineCommands.y});
    const auto commandsOrigin = ImGui::GetCursorScreenPos();
    auto* separators = ImGui::GetWindowDrawList();
    const float dividerX = commandsOrigin.x + layout.pipelineCommands.width + panelStyle.ItemSpacing.x * 0.5f;
    separators->AddLine({dividerX, commandsOrigin.y},
        {dividerX, commandsOrigin.y + layout.pipelineCommands.height}, ImGui::GetColorU32(ImGuiCol_Separator));
    if (ImGui::BeginChild("##PipelineCommands",
        {layout.pipelineCommands.width, layout.pipelineCommands.height}, ImGuiChildFlags_AlwaysUseWindowPadding))
    {
        // Match the borderless pipeline panel's header origin while keeping
        // horizontal padding in the command column.
        ImGui::SetCursorPosY(0);
        DrawPanelCaption("Manage Pipeline", _fontRegistry.bold);
        if (ImGui::Button(layout.pipelineCollapsed ? "Expand pipeline" : "Collapse pipeline", {-1, 0}))
        {
            _layoutPreferences.pipelineCollapsed = !_layoutPreferences.pipelineCollapsed;
            _lastPipelineSelection = -1;
            SaveLayoutPreferences();
        }
        UI::Button("Clear pipeline", {-1, 0}, [this] { ClearPipeline(); }, "Clear all processing steps");
        ImGui::Spacing();
        ImGui::TextDisabled("Pipeline help (?)");
        UI::SetTooltipIfHovered("Double-click a card to select it.\nHold Shift and use the mouse wheel to scroll horizontally.", cMenuScaling);
    }
    ImGui::EndChild();

    ImGui::SetCursorPos({layout.pipeline.x, layout.pipeline.y});
    const int count = GetPipelineSize();
    const int columns = std::max(count + 1, layout.visibleCells);
    if (BeginPipelinePanel("##PipelineSection", {layout.pipeline.width, layout.pipeline.height}, columns, layout.cellWidth))
    {
        const auto& style = ImGui::GetStyle();
        const float viewportWidth = ImGui::GetWindowWidth() - 2 * style.WindowPadding.x;
        if (layout.pipelineCollapsed)
        {
            const auto stripOrigin = ImGui::GetCursorScreenPos();
            const float inset = (layout.cellWidth - layout.cardSize.x) * 0.5f;
            if (!_lockInterface && ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows))
            {
                auto hit = layout;
                hit.pipeline = {0, 0, viewportWidth, layout.pipeline.height};
                ProcessSelectionMouse(_selection, count, hit,
                    {stripOrigin.x + ImGui::GetScrollX(), stripOrigin.y}, ImGui::GetScrollX(), 0);
            }
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + inset);
            auto element = _pPipelineHead;
            for (int index = 0; index < columns; ++index)
            {
                if (index > 0) ImGui::SameLine(0, layout.cellWidth - layout.cardSize.x);
                ImGui::PushID(index);
                const auto label = std::to_string(index + 1) + "  " +
                    (element ? element->GetDisplayName() : "+ Add step");
                ImGui::PushStyleVar(ImGuiStyleVar_SelectableTextAlign, {0.5f, 0.5f});
                if (ImGui::Selectable(label.c_str(), false, ImGuiSelectableFlags_None,
                    {layout.cardSize.x, ImGui::GetFrameHeight()}) && !_lockInterface)
                    UpdateActiveElement(index, true);
                ImGui::PopStyleVar();
                if (_selection.activeIndex == index && _selection.elementSelected)
                    ImGui::GetWindowDrawList()->AddRect(ImGui::GetItemRectMin(), ImGui::GetItemRectMax(), IM_COL32(255, 0, 0, 255));
                const int selectionKey = _selection.activeIndex * 2 + int(_selection.elementSelected);
                if (_selection.activeIndex == index && (_lastPipelineSelection != selectionKey || _lastPipelineWidth != viewportWidth))
                {
                    ImGui::SetScrollHereX(0.5f);
                    _lastPipelineSelection = selectionKey;
                    _lastPipelineWidth = viewportWidth;
                }
                ImGui::PopID();
                if (element) element = element->GetOutput();
            }
            if (!_selection.elementSelected)
            {
                const float x = stripOrigin.x + _selection.activeIndex * layout.cellWidth;
                ImGui::GetWindowDrawList()->AddLine({x, stripOrigin.y},
                    {x, stripOrigin.y + ImGui::GetFrameHeight()}, IM_COL32(255, 0, 0, 255), 2);
            }
            ImGui::EndChild();
            return;
        }
        const float insetX = (layout.cellWidth - layout.cardSize.x) * 0.5f;
        const float insetY = style.WindowPadding.y;
        const float headerHeight = std::max(24.0f, ImGui::GetTextLineHeightWithSpacing() + style.ItemSpacing.y);
        const float scrollbarHeight = ImGui::GetScrollMaxX() > 0 ? style.ScrollbarSize : 0;
        const float cellHeight = layout.pipeline.height - 2 * style.WindowPadding.y - scrollbarHeight;
        const auto gridOrigin = ImGui::GetCursorScreenPos();
        auto* grid = ImGui::GetWindowDrawList();
        const auto borderColor = ImGui::GetColorU32(ImGuiCol_TableBorderStrong);
        grid->AddRect(gridOrigin, {gridOrigin.x + columns * layout.cellWidth, gridOrigin.y + cellHeight}, borderColor);
        grid->AddLine({gridOrigin.x, gridOrigin.y + headerHeight},
            {gridOrigin.x + columns * layout.cellWidth, gridOrigin.y + headerHeight}, borderColor);
        for (int column = 1; column < columns; ++column)
            grid->AddLine({gridOrigin.x + column * layout.cellWidth, gridOrigin.y},
                {gridOrigin.x + column * layout.cellWidth, gridOrigin.y + cellHeight}, borderColor);
        const int selectionKey = _selection.activeIndex * 2 + int(_selection.elementSelected);
        if (_lastPipelineSelection != selectionKey || _lastPipelineWidth != viewportWidth)
        {
            const float left = _selection.activeIndex * layout.cellWidth +
                (_selection.elementSelected ? insetX : 0);
            const float right = left + (_selection.elementSelected ? layout.cardSize.x : 2.0f);
            if (left < ImGui::GetScrollX())
                ImGui::SetScrollX(left);
            else if (right > ImGui::GetScrollX() + viewportWidth)
                ImGui::SetScrollX(right - viewportWidth);
            _lastPipelineSelection = selectionKey;
            _lastPipelineWidth = viewportWidth;
        }

        // Hit testing uses the same content coordinates as the scrolled cards.
        if (!_lockInterface && !_showHelpPopup &&
            ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows))
        {
            auto hitLayout = layout;
            hitLayout.pipeline = {0, 0, viewportWidth, cellHeight};
            ProcessSelectionMouse(_selection, count, hitLayout,
                {gridOrigin.x + ImGui::GetScrollX(), gridOrigin.y},
                ImGui::GetScrollX(), 0);
        }

        auto element = _pPipelineHead;
        const float height = std::min(layout.cardSize.y,
            cellHeight - headerHeight - 2 * insetY);
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + insetX);
        const float cellTop = ImGui::GetCursorPosY();
        for (int index = 0; index < columns; ++index)
        {
            if (index > 0)
                ImGui::SameLine(0, layout.cellWidth - layout.cardSize.x);
            ImGui::PushID(index);
            ImGui::BeginGroup();
            const auto label = std::to_string(index + 1) + (element ? "" : "  + Add step");
            ImGui::PushStyleVar(ImGuiStyleVar_SelectableTextAlign, {0.5f, 0.5f});

            ImGui::SetCursorPosY(ImGui::GetCursorPosY() + style.ItemSpacing.y);
            if (ImGui::Selectable(label.c_str(), false,
                ImGuiSelectableFlags_None, {layout.cardSize.x, headerHeight - 2 * style.ItemSpacing.y}) && !_lockInterface)
                UpdateActiveElement(index, true);
            ImGui::PopStyleVar();
            ImGui::SetCursorPosY(cellTop + headerHeight + insetY);
            const bool selected = _selection.activeIndex == index && _selection.elementSelected;
            ImGui::PushStyleColor(ImGuiCol_Border, selected ? ImVec4(1, 0, 0, 1) : ImVec4(0.38f, 0.40f, 0.45f, 1));
            ImGui::PushStyleVar(ImGuiStyleVar_ChildBorderSize, selected ? 2.0f : 1.0f);
            if (element)
            {
                element->DrawCompactCard({layout.cardSize.x, height});
                element = element->GetOutput();
            }
            else
            {
                ImGui::BeginChild("empty", {layout.cardSize.x, height}, ImGuiChildFlags_Borders);
                ImGui::TextWrapped("Select this position, then choose a tool.");
                ImGui::EndChild();
            }
            ImGui::PopStyleVar();
            ImGui::PopStyleColor();
            ImGui::EndGroup();
            ImGui::PopID();
        }
        if (!_selection.elementSelected)
        {
            const float x = gridOrigin.x + _selection.activeIndex * layout.cellWidth;
            grid->AddLine({x, gridOrigin.y}, {x, gridOrigin.y + cellHeight},
                IM_COL32(255, 0, 0, 255), 2.0f);
        }
    }
    ImGui::EndChild();
}

void MainWindow::DrawDialog()
{
    UpdateProcessing();
    if ( IsInterfaceLocked() && !ImGui::IsPopupOpen( "", ImGuiPopupFlags_AnyPopupId ) && !FileDialog::Instance().IsOpened() && !_processingState.IsBusy() )
        UnlockInterface();

    if ( !_lockInterface && !_showHelpPopup && !ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel) )
    {
        ProcessKeyboardEvents();
    }

    DrawToolbar();
    DrawAdaptivePanels(GetPanelLayout());
    DrawMenu();

    ImGui::PushFont( _fontRegistry.byDefault );

    const auto layout = GetPanelLayout();
    _visibleCellsCount = layout.visibleCells;
    auto drawList = ImGui::GetWindowDrawList();
    const auto windowPos = ImGui::GetWindowPos();
    const ImVec2 origin{windowPos.x - ImGui::GetScrollX(), windowPos.y - ImGui::GetScrollY()};
    auto pElement = GetActiveElement();    
    const auto imageRegionAvail = GetImageRegionAvail();   

    if (layout.frameControls.width > 0 && layout.frameControls.height > 0)
    {
        ImGui::SetCursorPos({layout.frameControls.x, layout.frameControls.y});
        if (ImGui::BeginChild("##StepInspector", {layout.frameControls.width, layout.frameControls.height},
            ImGuiChildFlags_Borders, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse))
        {
            DrawInspectorContent();
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

            const auto fitted = layout.FitPreview({float(width), float(height)});
            if (fitted.width > 0 && fitted.height > 0)
            {
                ImGui::SetCursorPos({fitted.x, fitted.y});
                ImGui::Image(pTexture->GetTexture(), {fitted.width, fitted.height});
                const ImVec2 screenTopLeft{origin.x + fitted.x, origin.y + fitted.y};
                pElement->DrawOnPreviewImage(drawList, screenTopLeft, {fitted.width, fitted.height});
            }
        }
    }    

    DrawKeyframes(layout);
    DrawPipeline(layout);
    if (layout.pipelineProgress.width > 0 && layout.pipelineProgress.height > 0)
    {
        ImGui::SetCursorPos({layout.pipelineProgress.x, layout.pipelineProgress.y});
        ImGui::ProgressBar(PipelineHeadProgress(_pPipelineHead.get()),
            {layout.pipelineProgress.width, layout.pipelineProgress.height});
    }
    DrawSplitters(layout);
    for (auto element = _pPipelineHead; element; element = element->GetOutput())
        element->DrawAuxiliaryUi();

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
        style.WindowPadding, ImGui::GetTextLineHeightWithSpacing(), _layoutPreferences);
}

void MainWindow::Show()
{
    const auto layout = GetPanelLayout();
    const auto padding = ImGui::GetStyle().WindowPadding;
    ImGui::SetNextWindowPos(_pos, ImGuiCond_Always);
    ImGui::SetNextWindowSize(_size, ImGuiCond_Always);
    if (layout.canvasSize.x > 0 && layout.canvasSize.y > 0)
        ImGui::SetNextWindowContentSize({layout.canvasSize.x - 2 * padding.x,
            layout.canvasSize.y - 2 * padding.y});
    if (DrawHeader())
        DrawDialog();
    ImGui::End();
}

ACMB_GUI_NAMESPACE_END
