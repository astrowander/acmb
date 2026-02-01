#pragma once
#include "PipelineElementWindow.h"
#include "./../Geometry/size.h"
#include "./../Geometry/point.h"

#include <array>
#include <unordered_map>
#include <chrono>

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

    size_t _visibleCellsCount = 0;
    size_t _firstVisibleElement = 0;
    size_t _activeElement = 0;
    bool _isElementSelected = true;

    bool _isBusy = false;
    bool _showResultsPopup = false;
    bool _showHelpPopup = false;

    bool _lockInterface = false;
    bool _enableCuda = false;

    std::chrono::time_point<std::chrono::high_resolution_clock> _startTime;
    std::string _durationString;

    SizeF _gridCellSize = { 1.0f, 1.0f };

    MainWindow( const ImVec2& pos, const ImVec2& size, const FontRegistry& fontRegistry );
    MainWindow( const MainWindow& ) = delete;
    MainWindow( MainWindow&& ) = delete;
    MainWindow& operator=( const MainWindow& ) = delete;
    MainWindow& operator=( MainWindow&& ) = delete;

    virtual void DrawDialog() override;

    void DrawMenu();

    void ProcessKeyboardEvents();
    void ProcessMouseEvents();

    virtual ImGuiWindowFlags flags() const override { return  ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoDecoration; }

    const FontRegistry& _fontRegistry;
    
    void OpenProject();
    void SaveProject();

    size_t GetPipelineSize() const { return _pPipelineHead ? _pPipelineHead->GetElementsCount() : 0; }

    PipelineElementWindow* GetActiveElement() const;

    void UpdateActiveElement(size_t newActiveElement);

public:
    virtual void SetSize(const ImVec2& size) override;
    virtual void Show() override;
    static MainWindow& GetInstance( const FontRegistry& fontRegistry = FontRegistry::Instance() );

    template<class ElementType>
    void AddElementToGrid( size_t posInPipeline, bool replace = false )
    {
        std::shared_ptr<PipelineElementWindow> pNode = _pPipelineHead;
        std::shared_ptr<PipelineElementWindow> pPrev = nullptr;

        for ( size_t i = 0; i < posInPipeline; ++i )
        {
            pPrev = pNode;
            pNode = pNode->GetOutput();
        }

        auto pNewElement = std::make_shared<ElementType>();

        if ( !pNode ) // insert at the end
        {
            if ( pPrev )
            {
                pPrev->SetOutput(pNewElement);
                pNewElement->SetInput(pPrev);
            }
            else
                _pPipelineHead = pNewElement;

            return;
        }

        auto pPrevElement = pNode->GetInput();
        auto pNextElement = pNode->GetOutput();

        if ( pPrevElement )
        {
            pNewElement->SetInput(pPrevElement);
            pPrevElement->SetOutput(pNewElement);
        }
        else
        {
            _pPipelineHead = pNewElement;
        }

        if ( !replace )
        {
            pNewElement->SetOutput(pNode);
            pNode->SetInput(pNewElement);
            return;
        }

        pNewElement->SetOutput(pNextElement);
        if ( pNextElement )
            pNextElement->SetInput(pNewElement);
    }

    void LockInterface() {
        _lockInterface = true;
    }

    void UnlockInterface() {
        _lockInterface = false;
    }

    bool IsInterfaceLocked() { return _lockInterface; }

    bool isCudaEnabled() { return _enableCuda; }

    void ClearPipeline()
    {
        _pPipelineHead.reset();
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
