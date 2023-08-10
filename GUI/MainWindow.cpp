#include "MainWindow.h"
#include "ImageReaderWindow.h"
#include "ImageWriterWindow.h"
#include "ConverterWindow.h"
#include "FontRegistry.h"

#include <thread>

ACMB_GUI_NAMESPACE_BEGIN

static constexpr int windowWidth = 1920;
static constexpr int windowHeight = 1080;

static constexpr int cHeadRowHeight = 25;
static constexpr int cGridTop = 150;
static constexpr int cGridLeft = 30;

static constexpr int cGridCellWidth = PipelineElementWindow::cElementWidth + 50;
static constexpr int cGridCellHeight = PipelineElementWindow::cElementHeight + 50;

MainWindow::MainWindow( const ImVec2& pos, const ImVec2& size, const FontRegistry& fontRegistry )
: Window( "acmb", size )
, _fontRegistry( fontRegistry )
{
    SetPos( pos );
    _viewportSize = { ( windowWidth - cGridLeft ) / cGridCellWidth, ( windowHeight - cGridTop ) / cGridCellHeight };
}

constexpr uint32_t U32Color( uint8_t r, uint8_t g, uint32_t b, uint32_t a )
{
    return r | ( g << 8 ) | ( b << 16 ) | ( a << 24 );
}
enum class UIColor : ImU32
{
    Arrow = U32Color( 0, 255, 0, 255 ),
    EmptyCell = U32Color( 32, 32, 32, 255 ),
    ActiveCellBorder = U32Color( 255, 0, 0, 255),
    TableBorders = U32Color( 64, 64, 64, 255 )
};

void MainWindow::ProcessKeyboardEvents()
{
    if ( ImGui::IsKeyPressed( ImGuiKey_LeftArrow ) )
    {
        _activeCell.x = std::max<int>( _activeCell.x - 1, 0 );
        if ( _activeCell.x < _viewportStart.x )
            _viewportStart.x = _activeCell.x;
    }

    if ( ImGui::IsKeyPressed( ImGuiKey_RightArrow ) )
    {
        _activeCell.x = std::min<int>( _activeCell.x + 1, cGridSize.width - 1 );
        if ( _activeCell.x >= _viewportStart.x + _viewportSize.width )
            _viewportStart.x = _activeCell.x - _viewportSize.width + 1;
    }

    if ( ImGui::IsKeyPressed( ImGuiKey_UpArrow ) )
    {
        _activeCell.y = std::max<int>( _activeCell.y - 1, 0 );
        if ( _activeCell.y < _viewportStart.y )
            _viewportStart.y = _activeCell.y;
    }

    if ( ImGui::IsKeyPressed( ImGuiKey_DownArrow ) )
    {
        _activeCell.y = std::min<int>( _activeCell.y + 1, cGridSize.height - 1 );
        if ( _activeCell.y >= _viewportStart.y + _viewportSize.height )
            _viewportStart.y = _activeCell.y - _viewportSize.height + 1;
    }

    const size_t gridIdx = _activeCell.y * cGridCellWidth + _activeCell.x;
    if ( ImGui::IsKeyPressed( ImGuiKey_Delete ) )
    {
        if ( _activeCell.x < cGridCellWidth - 1 && _grid[gridIdx + 1] )
            _grid[gridIdx + 1]->SetLeftInput( nullptr );

        if ( _writers.contains( gridIdx ) )
            _writers.erase( gridIdx );

        _grid[gridIdx].reset();
    }
}

void MainWindow::DrawDialog()
{
    ProcessKeyboardEvents();
    ImGui::NewLine();
    ImGui::SetCursorPosX( cGridLeft );

    ImGui::PushFont( _fontRegistry.iconsFont );
    if ( ImGui::Button( "\xef\x87\x85", {50, 50}) )
    {
        AddElementToGrid<ImageReaderWindow>( _activeCell );
    }
    ImGui::SameLine();
    if ( ImGui::Button( "\xef\x86\xb8", { 50, 50 } ) )
    {
        AddElementToGrid<ConverterWindow>( _activeCell );
    }
    ImGui::SameLine();
    if ( ImGui::Button( "\xef\x83\x87", { 50, 50 } ) )
    {
        AddElementToGrid<ImageWriterWindow>( _activeCell );
    }

    ImGui::PopFont();

    if ( ImGui::Button( "Run" ) )
    {
        _errors.clear();
        _isBusy = true;
        std::thread process( [&]
        {
            if ( _writers.empty() )
                _errors.emplace_back( "No writers" );

            for ( auto pWriter : _writers )
            {
                const auto errors = pWriter.second.lock()->RunAllTasks();
                _errors.insert( _errors.end(), errors.begin(), errors.end() );
            }

            _isBusy = false;
            _finished = true;
        } );
        process.detach();
    }

    ImGui::SetCursorPos( { 0, cGridTop - cHeadRowHeight } );
    if ( ImGui::Button( "##ClearTable", { cGridLeft, cHeadRowHeight } ) )
    {
        for ( auto& pElement : _grid )
            pElement.reset();
    }

    //ImGui::BeginChild( "GridWindow" );
    auto drawList = ImGui::GetWindowDrawList();

    drawList->AddLine( { 0, cGridTop - cHeadRowHeight - 1 }, { _size.x, cGridTop - cHeadRowHeight - 1 }, ImU32( UIColor::TableBorders ) );
    drawList->AddLine( { 1, cGridTop - cHeadRowHeight - 1 }, { 1, _size.y }, ImU32( UIColor::TableBorders ), 2.0f );

    ImVec2 topLeft;
    ImVec2 bottomRight;

    for ( int x = 0; x < int( _viewportSize.width ); ++x )
    {
        topLeft.x = float( cGridLeft + x * cGridCellWidth );
        drawList->AddLine( { topLeft.x - 1, cGridTop - cHeadRowHeight - 1 }, { topLeft.x - 1, _size.y }, ImU32( UIColor::TableBorders ) );
        ImGui::SetCursorPos( { topLeft.x + cGridCellWidth * 0.5f, cGridTop - cHeadRowHeight + ImGui::GetTextLineHeightWithSpacing() * 0.25f });

        std::string columnHeader( 1, 'A' + x + _viewportStart.x );
        ImGui::Text( "%s", columnHeader.c_str() );
    }

    for ( int y = 0; y < int( _viewportSize.height ); ++y )
    {
        topLeft.y = float( cGridTop + y * cGridCellHeight );
        bottomRight.y = topLeft.y + cGridCellHeight;
        
        drawList->AddLine( { 0, topLeft.y - 1 }, { _size.x, topLeft.y - 1 }, ImU32( UIColor::TableBorders ) );
        ImGui::SetCursorPos( { cGridLeft * 0.5f, topLeft.y + cGridCellHeight * 0.5f - ImGui::GetTextLineHeightWithSpacing() * 0.5f } );
        std::string rowHeader = std::to_string( y + _viewportStart.y + 1 );
        ImGui::Text( "%s", rowHeader.c_str() );


        for ( int x = 0; x < int( _viewportSize.width ); ++x )
        {
            topLeft.x = float( cGridLeft + x * cGridCellWidth );
            bottomRight.x = topLeft.x + cGridCellWidth;

            const Point gridPos = { x + _viewportStart.x, y + _viewportStart.y };
            const size_t gridIdx = gridPos.y * cGridSize.width + gridPos.x;

            ImGui::PushClipRect( { topLeft.x - 1, topLeft.y - 1 }, bottomRight, false );

            if ( _activeCell == gridPos )
                drawList->AddRect( { topLeft.x + 24, topLeft.y + 24 }, { bottomRight.x - 24, bottomRight.y - 24 }, ImU32( UIColor::ActiveCellBorder ), 0, 0, 2.0f );

            if ( !_grid[gridIdx] )
            {
                drawList->AddRectFilled( { topLeft.x + 24, topLeft.y + 24 }, { bottomRight.x - 24, bottomRight.y - 24 }, ImU32( UIColor::EmptyCell ) );
                if ( _activeCell == gridPos )
                    drawList->AddRect( { topLeft.x + 24, topLeft.y + 24 }, { bottomRight.x - 24, bottomRight.y - 24 }, ImU32( UIColor::ActiveCellBorder ), 0, 0, 2.0f );
                ImGui::PopClipRect();
                continue;
            }

            _grid[gridIdx]->SetPos( { topLeft.x + 25.0f, topLeft.y + 25.0f } );

            if ( gridPos.x < int( cGridSize.width - 1 ) && _grid[gridIdx + 1] && _grid[gridIdx + 1]->GetLeftInput() == _grid[gridIdx] )
            {
                drawList->AddRectFilled( { bottomRight.x - 25.0f, topLeft.y + cGridCellHeight * 0.5f - 25.0f },
                                         { bottomRight.x, topLeft.y + cGridCellHeight * 0.5f + 25.0f },
                                         ImU32( UIColor::Arrow ) );
            }

            if ( gridPos.x > 0 && _grid[gridIdx - 1] && _grid[gridIdx]->GetLeftInput() == _grid[gridIdx - 1] )
            {
                drawList->AddTriangleFilled( { topLeft.x, topLeft.y + cGridCellHeight * 0.5f - 50.0f },
                                             { topLeft.x + 25.0f, topLeft.y + cGridCellHeight * 0.5f },
                                             { topLeft.x, topLeft.y + cGridCellHeight * 0.5f + 50.0f },
                                             ImU32( UIColor::Arrow )
                );
            }

            if ( gridPos.y < int( cGridSize.height - 1 ) && _grid[gridIdx + cGridSize.width] )
            {
                //draw arrow to the up
            }

            if ( _activeCell == gridPos )
                drawList->AddRect( { topLeft.x + 24, topLeft.y + 24 }, { bottomRight.x - 24, bottomRight.y - 24 }, ImU32( UIColor::ActiveCellBorder ), 0, 0, 2.0f );

            ImGui::PopClipRect();
        }
    }
    
    //ImGui::EndChild();

    if ( _finished )
    {
        ImGui::OpenPopup( "ResultsPopup" );
        _finished = false;
    }

    if ( ImGui::BeginPopup( "ResultsPopup" ) )
    {
        if ( _errors.empty() )
        {
            ImGui::TextColored( { 0, 1, 0, 1 }, "Success!" );
            return ImGui::EndPopup();
        }

        for ( const auto& error : _errors )
        {
            ImGui::TextColored( { 1, 0, 0, 1 }, "%s", error.c_str() );
        }

        return ImGui::EndPopup();
    }
}

std::shared_ptr<MainWindow> MainWindow::Create( const FontRegistry& fontRegistry )
{

    //const auto viewport = ImGui::GetMainViewport();
    //ImGui::GetIO().Fonts->AddFontFromFileTTF( "Fonts/fa-solid-900.ttf", 32 );
    return std::shared_ptr<MainWindow>( new MainWindow( { 0, 0 }, { windowWidth, windowHeight }, fontRegistry ) );
}

template<class ElementType>
std::string MainWindow::AddElementToGrid( const Point& pos )
{
    auto pElement = std::make_shared<ElementType>( pos );

    assert( pos.x < cGridSize.width && pos.y < cGridSize.height );

    const size_t ind = pos.y * cGridSize.width + pos.x;

    const auto pLeft = pos.x > 0 ? _grid[ind - 1] : nullptr;
    const auto pTop = pos.y > 0 ? _grid[ind - cGridSize.width] : nullptr;
    const auto pRight = pos.x < cGridSize.width - 1 ? _grid[ind + 1] : nullptr;
    const auto pBottom = pos.y < cGridSize.width - 1 ? _grid[ind + cGridSize.width] : nullptr;

    const auto pBottomRight = ( pos.y < cGridSize.height - 1 ) && ( pos.x < cGridSize.width - 1 ) ? _grid[ind + cGridSize.width + 1] : nullptr;
    const auto pBottomLeft = ( pos.y < cGridSize.height - 1 ) && ( pos.x > 0 ) ? _grid[ind + cGridSize.width - 1] : nullptr;
    const auto pTopLeft = ( pos.y > 0 ) && ( pos.x > 0 ) ? _grid[ind - cGridSize.width - 1] : nullptr;

    const auto flags = pElement->GetInOutFlags();

    using PEFlags = PipelineElementWindow::RequiredInOutFlags;
    if ( ( flags & PEFlags::NoInput ) && ( pLeft || pBottom ) )
        return "This element cannot have inputs";

    if ( ( flags & PEFlags::StrictlyOneInput ) && pLeft && pBottom )
        return "This element can have strictly one input";

    if ( pLeft )
    {
        if ( pLeft->GetInOutFlags() & PEFlags::NoOutput )
            return "Left element cannot have outputs";

        if ( pBottomLeft )
            return "Left element already have an output";
    }

    if ( pBottom )
    {
        if ( pBottom->GetInOutFlags() & PEFlags::NoOutput )
            return "Bottom element cannot have outputs";

        if ( pBottomRight )
            return "Bottom element already have an output";
    }

    if ( pTop )
    {
        const auto topFlags = pTop->GetInOutFlags();
        if ( ( topFlags & PEFlags::NoInput ) )
            return "Top element cannot have inputs";

        if ( ( topFlags & PEFlags::StrictlyOneInput ) && pTopLeft )
            return "Top element can have strictly one input";
    }

    if ( pRight )
    {
        const auto rightFlags = pRight->GetInOutFlags();
        if ( ( rightFlags & PEFlags::NoInput ) )
            return "Right element cannot have inputs";

        if ( ( rightFlags & PEFlags::StrictlyOneInput ) && pBottomRight )
            return "Right element can have strictly one input";
    }

    if ( pLeft )
        pElement->SetLeftInput( pLeft );

    if ( pBottom )
        pElement->SetBottomInput( pBottom );

    if ( pRight )
        pRight->SetLeftInput( pElement );

    if ( pTop )
        pTop->SetBottomInput( pElement );

    _grid[ind] = pElement;

    if constexpr ( std::is_same_v<ElementType, ImageWriterWindow> )
        _writers.insert_or_assign( ind, std::static_pointer_cast< ImageWriterWindow >( pElement ) );

    return {};
}

void MainWindow::Show()
{
    Window::Show();

    for ( size_t i = 0; i < _grid.size(); ++i )
    {
        const size_t x = i % cGridSize.width;
        const size_t y = i / cGridSize.width;

        if ( x < _viewportStart.x || x > _viewportStart.x + _viewportSize.width ||
             y < _viewportStart.y || y > _viewportStart.y + _viewportSize.height )
        {
            continue;
        }

        if ( _grid[i] )
            _grid[i]->Show();
    }
}

ACMB_GUI_NAMESPACE_END


