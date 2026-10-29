#pragma once
#include "PipelineElementWindow.h"
#include "MainWindowLayout.h"
#include "PipelineEditing.h"
#include "GuiInteractions.h"
#include "./../Geometry/size.h"
#include "./../Geometry/point.h"

#include <array>
#include <unordered_map>
#include <chrono>
#include <atomic>
#include <future>

#ifdef _WIN32
#include <d3d11.h>
#undef min
#undef max
#elif defined ( __linux__ )
#include "imgui_impl_vulkan.h"
#endif // _WIN32

ACMB_GUI_NAMESPACE_BEGIN

class PipelineElementWindow;
class ImageWriterWindow;
struct FontRegistry;
class FileDialog;

class MainWindow : public Window
{
private:
    //std::unordered_map<size_t, std::weak_ptr<ImageWriterWindow>> _writers;
    std::vector<std::string> _errors;    
    //Size _actualGridSize = {};
    std::shared_ptr<PipelineElementWindow> _pPipelineHead;

    int _visibleCellsCount = 0;
    PipelineSelection _selection;
    int _lastPipelineSelection = -1;
    float _lastPipelineWidth = 0;

    bool _showResultsPopup = false;
    bool _showHelpPopup = false;

    bool _lockInterface = false;
    bool _enableCuda = false;

    std::chrono::time_point<std::chrono::high_resolution_clock> _startTime;
    std::string _durationString;

    SizeF _gridCellSize = { 1.0f, 1.0f };

    ProcessingState _processingState;
    // Declared last so destruction waits for processing before releasing state.
    std::future<std::vector<std::string>> _processingFuture;

    MainWindow( const ImVec2& pos, const ImVec2& size, const FontRegistry& fontRegistry );
    MainWindow( const MainWindow& ) = delete;
    MainWindow( MainWindow&& ) = delete;
    MainWindow& operator=( const MainWindow& ) = delete;
    MainWindow& operator=( MainWindow&& ) = delete;

    virtual void DrawDialog() override;

    void DrawMenu();
    void DrawToolbar();
    void DrawPipeline(const MainWindowLayout& layout);
    MainWindowLayout GetPanelLayout() const;

    void ProcessKeyboardEvents();

    virtual ImGuiWindowFlags flags() const override { return ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_HorizontalScrollbar; }

    const FontRegistry& _fontRegistry;
    
    void OpenProject();
    bool LoadProject( const std::string& filePath );
    void SaveProject();
    void WriteProject( const std::string& filePath ) const;
    bool AutoSaveProject();
    void FinishProcessing();
    void UpdateProcessing();

    int GetPipelineSize() const { return _pPipelineHead ? _pPipelineHead->GetElementsCount() : 0; }

    PipelineElementWindow* GetActiveElement() const;

    void UpdateActiveElement(int newActiveElement, bool isElementSelected);

public:
    void RestoreAutoSavedProject();
    bool PrepareForExit();
    virtual void SetSize(const ImVec2& size) override;
    virtual void Show() override;
    static MainWindow& GetInstance( const FontRegistry& fontRegistry = FontRegistry::Instance() );

    template<class ElementType>
    void AddElementToGrid( size_t posInPipeline, bool replace = false )
    {
        InsertPipelineElement<PipelineElementWindow>(_pPipelineHead,
            std::make_shared<ElementType>(), posInPipeline, replace);
        UpdateActiveElement(_selection.activeIndex, _selection.elementSelected);
    }

    void LockInterface() {
        _lockInterface = true;
    }

    void UnlockInterface() {
        _lockInterface = false;
    }

    bool IsInterfaceLocked() { return _lockInterface; }

    bool IsProcessingCancelled() const { return _processingState.IsCancellationRequested(); }

    bool isCudaEnabled() { return _enableCuda; }

    void ClearPipeline()
    {
        ClearEditedPipeline(_pPipelineHead, _selection);
        _lastPipelineSelection = -1;
    }

    RectF GetImageRegionAvail() const;

#ifdef _WIN32
private:
    ID3D11Device* _pD3D11Device = nullptr;
public:
    ID3D11Device* GetD3D11Device() { return _pD3D11Device; }
    void SetD3D11Device( ID3D11Device* pDevice ) { _pD3D11Device = pDevice; }
#elif defined ( __linux__ )
private:
    VkPhysicalDevice _physicalDevice;
    VkDevice _device;
    VkAllocationCallbacks* _allocator;
    ImGui_ImplVulkanH_Window* _mainWindowData;
    VkQueue _queue;
public:
    VkPhysicalDevice GetPhysicalDevice() { return _physicalDevice; }
    void SetPhysicalDevice( VkPhysicalDevice physicalDevice) { _physicalDevice = physicalDevice; }
    VkDevice GetDevice() { return _device; }
    void SetDevice( VkDevice device) { _device = device; }
    VkAllocationCallbacks* GetAllocator() {return _allocator;}
    void SetAllocator(VkAllocationCallbacks* allocator) { _allocator = allocator;}
    ImGui_ImplVulkanH_Window* GetMainWindowData() { return _mainWindowData; }
    void SetMainWindowData( ImGui_ImplVulkanH_Window* val ) { _mainWindowData = val; }
    VkQueue GetQueue() {return _queue;}
    void SetQueue( VkQueue queue ) { _queue = queue;}
#endif
};

ACMB_GUI_NAMESPACE_END
