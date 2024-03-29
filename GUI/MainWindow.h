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
public:
    static constexpr Size cGridSize = { 26, 26 };
private:
    std::unordered_map<size_t, std::weak_ptr<ImageWriterWindow>> _writers;
    std::vector<std::string> _errors;    
    Size _actualGridSize = {};
    std::array<std::shared_ptr< PipelineElementWindow>, cGridSize.width * cGridSize.height> _grid;

    Size _viewportSize;
    Point _viewportStart;
    Point _activeCell;

    bool _isBusy = false;
    bool _showResultsPopup = false;
    bool _showHelpPopup = false;

    bool _lockInterface = false;
    bool _enableCuda = false;

    std::chrono::time_point<std::chrono::high_resolution_clock> _startTime;
    std::string _durationString;

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

public:

    virtual void Show() override;
    static MainWindow& GetInstance( const FontRegistry& fontRegistry = FontRegistry::Instance() );

    template<class ElementType>
    void AddElementToGrid( const Point& pos )
    {
        const size_t ind = pos.y * cGridSize.width + pos.x;

        if ( _writers.contains( ind ) )
            _writers.erase( ind );

        auto pElement = std::make_shared<ElementType>( pos );

        assert( pos.x < cGridSize.width && pos.y < cGridSize.height );        

        const auto pLeft = pos.x > 0 ? _grid[ind - 1] : nullptr;
        const auto pTop = pos.y > 0 ? _grid[ind - cGridSize.width] : nullptr;
        const auto pRight = pos.x < cGridSize.width - 1 ? _grid[ind + 1] : nullptr;
        const auto pBottom = pos.y < cGridSize.width - 1 ? _grid[ind + cGridSize.width] : nullptr;

        const auto flags = pElement->GetInOutFlags();

        if ( pElement->HasFreeInputs() && pLeft && pLeft->HasFreeOutputs() )
        {
            pLeft->SetRightOutput( pElement );
            pElement->SetLeftInput( pLeft );
            pElement->SetLeftRelationType( pLeft->GetRightRelationType() );
        }

        if ( pElement->HasFreeOutputs() && pRight && pRight->HasFreeInputs() )
        {
            pRight->SetLeftInput( pElement );
            pElement->SetRightOutput( pRight );
            pElement->SetRightRelationType( pRight->GetLeftRelationType() );
        }

        if ( pElement->HasFreeInputs() && pTop && pTop->HasFreeOutputs() )
        {
            pTop->SetBottomOutput( pElement );
            pElement->SetTopInput( pTop );
            pElement->SetTopRelationType( pTop->GetBottomRelationType() );
        }

        if ( pElement->HasFreeOutputs() && pBottom && pBottom->HasFreeInputs() )
        {
            pBottom->SetTopInput( pElement );
            pElement->SetBottomOutput( pBottom );
            pElement->SetBottomRelationType( pBottom->GetTopRelationType() );
        }

        _grid[ind] = pElement;

        if constexpr ( std::is_same_v<ElementType, ImageWriterWindow> )
            _writers.insert_or_assign( ind, std::static_pointer_cast< ImageWriterWindow >( pElement ) );
    }

    void LockInterface() {
        _lockInterface = true;
    }

    void UnlockInterface() {
        _lockInterface = false;
    }

    bool IsInterfaceLocked() { return _lockInterface; }

    bool isCudaEnabled() { return _enableCuda; }

    void ClearTable()
    {
        _writers.clear();
        for ( auto& pElement : _grid )
            pElement.reset();
    }

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
