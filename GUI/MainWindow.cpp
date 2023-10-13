#include "MainWindow.h"
#include "ImageReaderWindow.h"
#include "ImageWriterWindow.h"
#include "ConverterWindow.h"
#include "ResizeWindow.h"
#include "StackerWindow.h"
#include "CropWindow.h"
#include "SubtractImageWindow.h"
#include "FlatFieldWindow.h"
#include "FontRegistry.h"

#include "ImGuiFileDialog/ImGuiFileDialog.h"
#include <fstream>
#include <thread>

ACMB_GUI_NAMESPACE_BEGIN

static constexpr int windowWidth = 1920;
static constexpr int windowHeight = 1080;

static constexpr int cHeadRowHeight = 25;
static constexpr int cGridTop = 150;
static constexpr int cGridLeft = 30;

static constexpr int cGridCellWidth = PipelineElementWindow::cElementWidth + 50;
static constexpr int cGridCellHeight = PipelineElementWindow::cElementHeight + 50;

static void SetTooltipIfHovered( const std::string& text, float scaling )
{
    if ( !ImGui::IsItemHovered() || ImGui::IsItemActive() )
        return;
    
    assert( scaling > 0.f );

    constexpr float cMaxWidth = 400.f;
    const auto& style = ImGui::GetStyle();
    auto textSize = ImGui::CalcTextSize( text.c_str(), nullptr, false, cMaxWidth * scaling - style.WindowPadding.x * 2 );
    ImGui::SetNextWindowSize( ImVec2{ textSize.x + style.WindowPadding.x * 2, 0 } );

    ImGui::BeginTooltip();
    ImGui::TextWrapped( "%s", text.c_str() );
    ImGui::EndTooltip();
}

MainWindow::MainWindow( const ImVec2& pos, const ImVec2& size, const FontRegistry& fontRegistry )
: Window( "acmb", size )
, _fontRegistry( fontRegistry )
{
    SetPos( pos );
    _viewportSize = { ( windowWidth - cGridLeft ) / cGridCellWidth, ( windowHeight - cGridTop ) / cGridCellHeight };

    MenuItemsHolder::GetInstance().AddItem( "Run", 1, "\xef\x81\x8B", "Run", [this] ( Point )
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
    } );

    MenuItemsHolder::GetInstance().AddItem( "File", 2, "\xef\x95\xad", "Save Project", [this]( Point )
    {
        auto pFileDialog = ImGuiFileDialog::Instance();
        pFileDialog->OpenDialog( "SaveProjectDialog", "Save Table", ".acmb", "", 1 );
        
    } );

    MenuItemsHolder::GetInstance().AddItem( "File", 1, "\xef\x81\xbc", "Open Project", [this] ( Point )
    {
        auto pFileDialog = ImGuiFileDialog::Instance();
        pFileDialog->OpenDialog( "OpenProjectDialog", "Load Table", ".acmb", "", 1 );
    } );


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

    const size_t gridIdx = _activeCell.y * cGridSize.width + _activeCell.x;
    if ( ImGui::IsKeyPressed( ImGuiKey_Delete ) )
    {
        if ( _activeCell.x < cGridSize.width - 1 && _grid[gridIdx + 1] )
            _grid[gridIdx + 1]->SetLeftInput( nullptr );

        if ( _writers.contains( gridIdx ) )
            _writers.erase( gridIdx );

        _grid[gridIdx].reset();
    }
}

void MainWindow::OpenProject( IGFD::FileDialog* pFileDialog )
{
    auto reportError = [this] ( const std::string msg )
    {
        _errors.push_back( msg );
        ImGui::OpenPopup( "ResultsPopup" );
        return ImGuiFileDialog::Instance()->Close();
    };
    std::string filePath = pFileDialog->GetFilePathName();
    std::ifstream fin( filePath, std::ios_base::in | std::ios_base::binary );

    _errors.clear();
    if ( !fin )
        return reportError( "Unable to open file" );

    fin.seekg( 0, std::ios_base::end );
    size_t streamSize = fin.tellg();
    if ( streamSize < 6 )
        return reportError( "File is too small" );

    fin.seekg( 0 );

    std::string header( 4, '\0' );
    fin.read( &header[0], 4 );
    if ( header != "ACMB" )
        return reportError( "File is corrupted" );

    Size actualGridSize;
    actualGridSize.width = fin.get();
    actualGridSize.height = fin.get();

    if ( actualGridSize.width > cGridSize.width || actualGridSize.height > cGridSize.height )
        return reportError( "Table is too large" );

    int charCount = actualGridSize.width * actualGridSize.height;
    if ( streamSize != charCount + 6 )
        return reportError( "File is corrupted" );

    std::string serialized( charCount, '\0' );
    fin.read( serialized.data(), charCount );
    DeserializeProject( serialized, actualGridSize );
}

void MainWindow::DeserializeProject( const std::string& serialized,  const Size& actualGridSize )
{
    for ( size_t i = 0; i < actualGridSize.height; ++i )
    for ( size_t j = 0; j < actualGridSize.width; ++j )
    {
        if ( uint8_t menuOrder = serialized[i * actualGridSize.width + j] )
            MenuItemsHolder::GetInstance().GetItems().at( "Tools" ).at( menuOrder )->action( Point{ .x = int( j ), .y = int( i ) } );
    }
}

void MainWindow::SaveProject( IGFD::FileDialog* pFileDialog )
{
    std::string filePath = pFileDialog->GetFilePathName();
    std::ofstream fout( filePath, std::ios_base::out | std::ios_base::binary );

    _errors.clear();
    if ( !fout )
    {
        _errors.push_back( "Unable to save file" );
        ImGui::OpenPopup( "ResultsPopup" );
        return ImGuiFileDialog::Instance()->Close();
    }

    fout.write( "ACMB", 4 );

    Size actualGridSize;

    for ( int i = 0; i < cGridSize.height; ++i )
        for ( int j = 0; j < cGridSize.width; ++j )
        {
            if ( _grid[i * cGridSize.width + j] )
            {
                if ( i >= actualGridSize.height )
                    actualGridSize.height = i + 1;

                if ( j >= actualGridSize.width )
                    actualGridSize.width = j + 1;
            }
        }

    fout.put( char( actualGridSize.width ) );
    fout.put( char( actualGridSize.height ) );

    std::string chars( actualGridSize.width * actualGridSize.height, '\0' );

    for ( size_t i = 0; i < actualGridSize.height; ++i )
        for ( size_t j = 0; j < actualGridSize.width; ++j )
        {
            chars[i * actualGridSize.width + j] = ( ( _grid[i * cGridSize.height + j] ) ? _grid[i * cGridSize.height + j]->GetMenuOrder() : 0 );
        }

    fout.write( chars.data(), chars.size() );
}

void MainWindow::DrawMenu()
{
    float shift = 0;

    for ( const auto& it : MenuItemsHolder::GetInstance().GetItems() )
    {
        const auto& category = it.first;
        const auto& items = it.second;

        const auto itemCount = items.size();
        const float menuWidth = itemCount * 50 + ( itemCount - 1 ) * ImGui::GetStyle().ItemSpacing.x;

        ImGui::BeginChild( category.c_str(), { menuWidth, 0});

        ImGui::PushFont( _fontRegistry.bold );
        ImGui::SeparatorText( category.c_str() );
        ImGui::PopFont();        

        for ( auto& item : items )
        {
            ImGui::PushFont( _fontRegistry.icons );

            if ( ImGui::Button( item.second->icon.c_str(), { 50, 50 } ) )
                item.second->action( _activeCell );

            ImGui::PopFont();
            
            SetTooltipIfHovered( item.second->tooltip, cMenuScaling );
            ImGui::SameLine();
        }

        ImGui::EndChild();
        
        ImGui::SameLine();
        ImGui::Dummy( { 2.0f * ImGui::GetStyle().ItemSpacing.x, 0 } );
        ImGui::SameLine();               
    }

    //ImGui::BeginChild( "EmptyMenuSpace", { -1, 0} );
    ImGui::PushFont( _fontRegistry.bold );
    ImGui::SeparatorText( "##EmptyMenuSpace");
    //ImGui::NewLine();
    //ImGui::Text( "BottomLine" );
    ImGui::PopFont();
    //ImGui::EndChild();


    auto pFileDialog = ImGuiFileDialog::Instance();
    if ( pFileDialog->Display( "OpenProjectDialog", {}, { 300 * cMenuScaling, 200 * cMenuScaling } ) )
    {
        if ( pFileDialog->IsOk() )
            OpenProject( pFileDialog );

        ImGuiFileDialog::Instance()->Close();
    }

    if ( pFileDialog->Display( "SaveProjectDialog", {}, { 300 * cMenuScaling, 200 * cMenuScaling } ) )
    {
        if ( pFileDialog->IsOk() )
            SaveProject( pFileDialog );

        ImGuiFileDialog::Instance()->Close();
    }
}

void MainWindow::DrawDialog()
{
    ProcessKeyboardEvents();
    DrawMenu();

    ImGui::PushFont( _fontRegistry.byDefault );

    ImGui::SetCursorPos( { 0, cGridTop - cHeadRowHeight } );
    if ( ImGui::Button( "##ClearTable", { cGridLeft, cHeadRowHeight } ) )
    {
        for ( auto& pElement : _grid )
            pElement.reset();
    }

    //ImGui::BeginChild( "GridWindow" );
    auto drawList = ImGui::GetWindowDrawList();
    
    drawList->AddLine( { 0, cGridTop - cHeadRowHeight - 20 }, { _size.x, cGridTop - cHeadRowHeight - 20 }, ImGui::GetColorU32( ImGui::GetStyleColorVec4( ImGuiCol_Separator ) ), 3.0f );

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

            if ( gridPos.y < int( cGridSize.height - 1 ) && _grid[gridIdx + cGridSize.width] && _grid[gridIdx]->GetBottomOutput() == _grid[gridIdx + cGridSize.width] )
            {
                drawList->AddRectFilled( { topLeft.x + cGridCellWidth * 0.5f - 25.0f, bottomRight.y - 25.0f },
                                         { topLeft.x + cGridCellWidth * 0.5f + 25.0f, bottomRight.y  },
                                         ImU32( UIColor::Arrow ) );
            }

            if ( gridPos.y > 0 && _grid[gridIdx - cGridSize.width] && _grid[gridIdx]->GetTopInput() == _grid[gridIdx - cGridSize.width] )
            {
                drawList->AddTriangleFilled( { topLeft.x + cGridCellWidth * 0.5f - 50.0f, topLeft.y },
                                             { topLeft.x + cGridCellWidth * 0.5f, topLeft.y + 25.0f },
                                             { topLeft.x + cGridCellWidth * 0.5f + 50.0f, topLeft.y },
                                             ImU32( UIColor::Arrow )
                );
            }

            if ( _activeCell == gridPos )
                drawList->AddRect( { topLeft.x + 24, topLeft.y + 24 }, { bottomRight.x - 24, bottomRight.y - 24 }, ImU32( UIColor::ActiveCellBorder ), 0, 0, 2.0f );

            ImGui::PopClipRect();
        }
    }

    ImGui::PopFont();
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

MainWindow& MainWindow::GetInstance( const FontRegistry& fontRegistry )
{
    static auto pInstance = std::unique_ptr<MainWindow>( new MainWindow( ImVec2{ 0, 0 }, ImVec2{ windowWidth, windowHeight }, fontRegistry ) );
    return *pInstance;
}



void MainWindow::Show()
{
    Window::Show();

    for ( size_t i = 0; i < _grid.size(); ++i )
    {
        const size_t x = i % cGridSize.width;
        const size_t y = i / cGridSize.width;

        if ( x < _viewportStart.x || x >= _viewportStart.x + _viewportSize.width ||
             y < _viewportStart.y || y >= _viewportStart.y + _viewportSize.height )
        {
            continue;
        }

        if ( _grid[i] )
            _grid[i]->Show();
    }
}

ACMB_GUI_NAMESPACE_END


