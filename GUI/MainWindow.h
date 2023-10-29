#pragma once
#include "PipelineElementWindow.h"
#include "./../Geometry/size.h"
#include "./../Geometry/point.h"
#include <array>
#include<unordered_map>

namespace IGFD
{
    class FileDialog;
}

ACMB_GUI_NAMESPACE_BEGIN

class PipelineElementWindow;
class ImageWriterWindow;
struct FontRegistry;

class MainWindow : public Window
{
    std::unordered_map<size_t, std::weak_ptr<ImageWriterWindow>> _writers;
    std::vector<std::string> _errors;

    static constexpr Size cGridSize = { 26, 26 };
    Size _actualGridSize = {};
    std::array<std::shared_ptr< PipelineElementWindow>, cGridSize.width * cGridSize.height> _grid;

    Size _viewportSize;
    Point _viewportStart;
    Point _activeCell;
    

    bool _isBusy = false;
    bool _finished = false;

    MainWindow( const ImVec2& pos, const ImVec2& size, const FontRegistry& fontRegistry );
    MainWindow( const MainWindow& ) = delete;
    MainWindow( MainWindow&& ) = delete;

    virtual void DrawDialog() override;

    void DrawMenu();
    //void DrawRunMenu();

    void ProcessKeyboardEvents();
    void ProcessMouseEvents();

    virtual ImGuiWindowFlags flags() override { return  ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoDecoration; }

    const FontRegistry& _fontRegistry;
    
    void OpenProject( IGFD::FileDialog* pFileDialog );
    void SaveProject( IGFD::FileDialog* pFileDialog );

   // std::pair<std::string, Size>

public:

    virtual void Show() override;
    static MainWindow& GetInstance( const FontRegistry& fontRegistry );

    template<class ElementType>
    void AddElementToGrid( const Point& pos )
    {
        auto pElement = std::make_shared<ElementType>( pos );

        assert( pos.x < cGridSize.width && pos.y < cGridSize.height );

        const size_t ind = pos.y * cGridSize.width + pos.x;

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
            pElement->SetRightRelationType( pLeft->GetLeftRelationType() );
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
};

ACMB_GUI_NAMESPACE_END