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
#include "FileDialog.h"
//#include "ImGuiFileDialog/ImGuiFileDialog.h"
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

static constexpr float cGridCellPadding = 25.0f;

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

    MenuItemsHolder::GetInstance().AddItem( "File", 2, "\xef\x95\xad", "Save Project", [this] ( Point )
    {
        FileDialog::Instance().OpenDialog( "SaveProjectDialog", "Save Table", ".acmb", "", 1 );
    } );

    MenuItemsHolder::GetInstance().AddItem( "File", 1, "\xef\x81\xbc", "Open Project", [this] ( Point )
    {
        FileDialog::Instance().OpenDialog( "OpenProjectDialog", "Load Table", ".acmb", "", 1 );
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
    ActiveCellBorder = U32Color( 255, 0, 0, 255 ),
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

void MainWindow::ProcessMouseEvents()
{
    if ( ImGui::IsMouseDoubleClicked( ImGuiMouseButton_Left ) )
    {
        auto mousePos = ImGui::GetMousePos();
        mousePos.x -= cGridLeft;
        mousePos.y -= cGridTop;

        if ( mousePos.x < 0 || mousePos.y < 0 )
            return;

        int col = int( mousePos.x ) / cGridCellWidth;
        int row = int( mousePos.y ) / cGridCellHeight;
        const auto gridIdx = row * cGridSize.width + col;
        const auto pElement = _grid[gridIdx];
        if ( !pElement )
            return;

        mousePos.x -= col * cGridCellWidth;
        mousePos.y -= row * cGridCellHeight;

        if ( auto pLeft = ( ( col > 0 ) ? _grid[gridIdx - 1] : nullptr ); pLeft && pElement->GetLeftInput() == pLeft && mousePos.x < cGridCellPadding )
        {
            const auto newRelationType = pElement->GetLeftRelationType() == PipelineElementWindow::RelationType::Batch ? PipelineElementWindow::RelationType::Join : PipelineElementWindow::RelationType::Batch;
            pElement->SetLeftRelationType( newRelationType );
            pLeft->SetRightRelationType( newRelationType );
        }

        if ( auto pRight = ( ( col < cGridSize.width - 1 ) ? _grid[gridIdx + 1] : nullptr ); pRight && pElement->GetRightOutput() == pRight && mousePos.x > cGridCellWidth - cGridCellPadding )
        {
            const auto newRelationType = pElement->GetRightRelationType() == PipelineElementWindow::RelationType::Batch ? PipelineElementWindow::RelationType::Join : PipelineElementWindow::RelationType::Batch;
            pElement->SetRightRelationType( newRelationType );
            pRight->SetLeftRelationType( newRelationType );
        }

        if ( auto pTop = ( ( row > 0 ) ? _grid[gridIdx - cGridSize.width] : nullptr ); pTop && pElement->GetTopInput() == pTop && mousePos.y < cGridCellPadding )
        {
            const auto newRelationType = pElement->GetTopRelationType() == PipelineElementWindow::RelationType::Batch ? PipelineElementWindow::RelationType::Join : PipelineElementWindow::RelationType::Batch;
            pElement->SetTopRelationType( newRelationType );
            pTop->SetBottomRelationType( newRelationType );
        }

        if ( auto pBottom = ( ( row < cGridSize.height - 1 ) ? _grid[gridIdx + cGridSize.width] : nullptr ); pBottom && pElement->GetBottomOutput() == pBottom && mousePos.y > cGridCellHeight - cGridCellPadding )
        {
            const auto newRelationType = pElement->GetBottomRelationType() == PipelineElementWindow::RelationType::Batch ? PipelineElementWindow::RelationType::Join : PipelineElementWindow::RelationType::Batch;
            pElement->SetBottomRelationType( newRelationType );
            pBottom->SetTopRelationType( newRelationType );
        }
    }
}

void MainWindow::OpenProject()
{
    for ( auto& pElement : _grid )
        pElement.reset();

    auto reportError = [this] ( const std::string msg )
    {
        _errors.push_back( msg );
        ImGui::OpenPopup( "ResultsPopup" );
        return FileDialog::Instance().Close();
    };
    std::string filePath = FileDialog::Instance().GetFilePathName();
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
    if ( streamSize < charCount + 6 )
        return reportError( "File is corrupted" );

    std::string serialized( charCount, '\0' );
    fin.read( serialized.data(), charCount );

    for ( size_t i = 0; i < actualGridSize.height; ++i )
    for ( size_t j = 0; j < actualGridSize.width; ++j )
    {
        if ( uint8_t menuOrder = serialized[i * actualGridSize.width + j] )
        {
            MenuItemsHolder::GetInstance().GetItems().at( "Tools" ).at( menuOrder )->action( Point{ .x = int( j ), .y = int( i ) } );
            if ( streamSize > charCount + 6 )
            {
                _grid[i * cGridSize.height + j]->Deserialize( fin );
                const char actualInputs = _grid[i * cGridSize.height + j]->GetActualInputs();

                if ( actualInputs & 1 )
                {
                    _grid[i * cGridSize.height + j]->SetLeftInput( _grid[i * cGridSize.height + j - 1] );
                    _grid[i * cGridSize.height + j - 1]->SetRightOutput( _grid[i * cGridSize.height + j] );
                }
                else if ( j > 0 )
                {
                    _grid[i * cGridSize.height + j]->SetLeftInput( nullptr );
                    if ( _grid[i * cGridSize.height + j - 1] )
                        _grid[i * cGridSize.height + j - 1]->SetRightOutput( nullptr );
                }

                if ( actualInputs & 2 )
                {
                    _grid[i * cGridSize.height + j]->SetTopInput( _grid[( i - 1 ) * cGridSize.height + j] );
                    _grid[( i - 1 ) * cGridSize.height + j]->SetBottomOutput( _grid[i * cGridSize.height + j] );
                }
                else if ( i > 0 )
                {
                    _grid[i * cGridSize.height + j]->SetTopInput( nullptr );
                    if ( _grid[( i - 1 ) * cGridSize.height + j] )
                        _grid[( i - 1 ) * cGridSize.height + j]->SetBottomOutput( nullptr );
                }
            }
        }
    }
}

void MainWindow::SaveProject()
{
    std::string filePath = FileDialog::Instance().GetFilePathName();
    std::ofstream fout( filePath, std::ios_base::out | std::ios_base::binary );

    _errors.clear();
    if ( !fout )
    {
        _errors.push_back( "Unable to save file" );
        ImGui::OpenPopup( "ResultsPopup" );
        return FileDialog::Instance().Close();
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

    for ( size_t i = 0; i < actualGridSize.height; ++i )
        for ( size_t j = 0; j < actualGridSize.width; ++j )
        {
            if ( _grid[i * cGridSize.height + j] )
                _grid[i * cGridSize.height + j]->Serialize( fout );
        }
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

        ImGui::BeginChild( category.c_str(), { menuWidth, 0 } );

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
    ImGui::SeparatorText( "##EmptyMenuSpace" );
    //ImGui::NewLine();
    //ImGui::Text( "BottomLine" );
    ImGui::PopFont();
    //ImGui::EndChild();


    auto fileDialog = FileDialog::Instance();
    if ( fileDialog.Display( "OpenProjectDialog", {}, { 300 * cMenuScaling, 200 * cMenuScaling } ) )
    {
        //MainWindowInterfaceLock lock;

        if ( fileDialog.IsOk() )
            OpenProject();

        fileDialog.Close();
    }

    if ( fileDialog.Display( "SaveProjectDialog", {}, { 300 * cMenuScaling, 200 * cMenuScaling } ) )
    {
        //MainWindowInterfaceLock lock;

        if ( fileDialog.IsOk() )
            SaveProject();

        fileDialog.Close();
    }
}

void MainWindow::DrawDialog()
{
    if ( !_lockInterface )
    {
        ProcessKeyboardEvents();
        ProcessMouseEvents();
    }

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
        ImGui::SetCursorPos( { topLeft.x + cGridCellWidth * 0.5f, cGridTop - cHeadRowHeight + ImGui::GetTextLineHeightWithSpacing() * 0.25f } );

        std::string columnHeader( 1, 'A' + x + _viewportStart.x );
        ImGui::Text( "%s", columnHeader.c_str() );
    }

    topLeft.x = float( cGridLeft + _viewportSize.width * cGridCellWidth );
    drawList->AddLine( { topLeft.x - 1, cGridTop - cHeadRowHeight - 1 }, { topLeft.x - 1, _size.y }, ImU32( UIColor::TableBorders ) );

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
                drawList->AddRect( { topLeft.x + cGridCellPadding - 1, topLeft.y + cGridCellPadding - 1 }, { bottomRight.x - cGridCellPadding + 1, bottomRight.y - cGridCellPadding + 1 }, ImU32( UIColor::ActiveCellBorder ), 0, 0, 2.0f );

            if ( !_grid[gridIdx] )
            {
                drawList->AddRectFilled( { topLeft.x + cGridCellPadding - 1, topLeft.y + cGridCellPadding - 1 }, { bottomRight.x - cGridCellPadding + 1, bottomRight.y - cGridCellPadding + 1 }, ImU32( UIColor::EmptyCell ) );
                if ( _activeCell == gridPos )
                    drawList->AddRect( { topLeft.x + cGridCellPadding - 1, topLeft.y + cGridCellPadding - 1 }, { bottomRight.x - cGridCellPadding + 1, bottomRight.y - cGridCellPadding + 1 }, ImU32( UIColor::ActiveCellBorder ), 0, 0, 2.0f );
                ImGui::PopClipRect();
                continue;
            }

            _grid[gridIdx]->SetPos( { topLeft.x + cGridCellPadding, topLeft.y + cGridCellPadding } );

            const float topArrowY = topLeft.y + cGridCellHeight * 0.5f - 2.0f * cGridCellPadding;
            const float centerArrowY = topLeft.y + cGridCellHeight * 0.5f;
            const float bottomArrowY = topLeft.y + cGridCellHeight * 0.5f + 2.0f * cGridCellPadding;
            
            if ( gridPos.x < int( cGridSize.width - 1 ) && _grid[gridIdx + 1] && _grid[gridIdx + 1]->GetLeftInput() == _grid[gridIdx] )
            {
                const float xStart = bottomRight.x - cGridCellPadding;
                drawList->AddLine( { xStart, topArrowY }, { bottomRight.x, topArrowY }, ImU32( UIColor::Arrow ), 3.0f * cMenuScaling );
                drawList->AddLine( { xStart, centerArrowY }, { bottomRight.x, centerArrowY }, ImU32( UIColor::Arrow ), 3.0f * cMenuScaling );
                drawList->AddLine( { xStart, bottomArrowY }, { bottomRight.x, bottomArrowY }, ImU32( UIColor::Arrow ), 3.0f * cMenuScaling );
            }

            if ( gridPos.x > 0 && _grid[gridIdx - 1] && _grid[gridIdx]->GetLeftInput() == _grid[gridIdx - 1] )
            {
                const float tipEnd = topLeft.x + cGridCellPadding;
                const float tipStart = tipEnd - 10.0f;

                if ( _grid[gridIdx]->GetLeftRelationType() == PipelineElementWindow::RelationType::Batch )
                {
                    drawList->AddLine( { topLeft.x, topArrowY }, { tipEnd, topArrowY }, ImU32( UIColor::Arrow ), 3.0f * cMenuScaling );
                    drawList->AddTriangleFilled( { tipStart, topArrowY - 10.0f }, { tipEnd, topArrowY }, { tipStart, topArrowY + 10.0f }, ImU32( UIColor::Arrow ) );

                    drawList->AddLine( { topLeft.x, centerArrowY }, { tipEnd, centerArrowY }, ImU32( UIColor::Arrow ), 3.0f * cMenuScaling );
                    drawList->AddTriangleFilled( { tipStart, centerArrowY - 10.0f }, { tipEnd, centerArrowY }, { tipStart, centerArrowY + 10.0f }, ImU32( UIColor::Arrow ) );

                    drawList->AddLine( { topLeft.x, bottomArrowY }, { tipEnd, bottomArrowY }, ImU32( UIColor::Arrow ), 3.0f * cMenuScaling );
                    drawList->AddTriangleFilled( { tipStart, bottomArrowY - 10.0f }, { tipEnd, bottomArrowY }, { tipStart, bottomArrowY + 10.0f }, ImU32( UIColor::Arrow ) );
                }
                else
                {
                    drawList->AddLine( { topLeft.x, topArrowY }, { tipEnd, centerArrowY }, ImU32( UIColor::Arrow ), 3.0f * cMenuScaling );
                    drawList->AddLine( { topLeft.x, centerArrowY }, { tipEnd, centerArrowY }, ImU32( UIColor::Arrow ), 3.0f * cMenuScaling );
                    drawList->AddLine( { topLeft.x, bottomArrowY }, { tipEnd, centerArrowY }, ImU32( UIColor::Arrow ), 3.0f * cMenuScaling );
                }
            }

            const float leftArrowX = topLeft.x + cGridCellWidth * 0.5f - 50.0f;
            const float centerArrowX = topLeft.x + cGridCellWidth * 0.5f;
            const float rightArrowX = topLeft.x + cGridCellWidth * 0.5f + 50.0f;

            if ( gridPos.y < int( cGridSize.height - 1 ) && _grid[gridIdx + cGridSize.width] && _grid[gridIdx]->GetBottomOutput() == _grid[gridIdx + cGridSize.width] )
            {
                const float yStart = bottomRight.y - cGridCellPadding;
                drawList->AddLine( { leftArrowX, yStart }, { leftArrowX, bottomRight.y }, ImU32( UIColor::Arrow ), 3.0f * cMenuScaling );
                drawList->AddLine( { centerArrowX, yStart }, { centerArrowX, bottomRight.y }, ImU32( UIColor::Arrow ), 3.0f * cMenuScaling );
                drawList->AddLine( { rightArrowX, yStart }, { rightArrowX, bottomRight.y }, ImU32( UIColor::Arrow ), 3.0f * cMenuScaling );
            }

            if ( gridPos.y > 0 && _grid[gridIdx - cGridSize.width] && _grid[gridIdx]->GetTopInput() == _grid[gridIdx - cGridSize.width] )
            {
                const float tipEnd = topLeft.y + cGridCellPadding;
                const float tipStart = tipEnd - 10.0f;

                if ( _grid[gridIdx]->GetTopRelationType() == PipelineElementWindow::RelationType::Batch )
                {
                    const float yStart = topLeft.y - 1.0f;
                    drawList->AddLine( { leftArrowX, yStart }, { leftArrowX, tipEnd }, ImU32( UIColor::Arrow ), 3.0f * cMenuScaling );
                    drawList->AddTriangleFilled( { leftArrowX - 10.0f, tipStart }, { leftArrowX, tipEnd }, { leftArrowX + 10.0f, tipStart }, ImU32( UIColor::Arrow ) );

                    drawList->AddLine( { centerArrowX, yStart }, { centerArrowX, tipEnd }, ImU32( UIColor::Arrow ), 3.0f * cMenuScaling );
                    drawList->AddTriangleFilled( { centerArrowX - 10.0f, tipStart }, { centerArrowX, tipEnd }, { centerArrowX + 10.0f, tipStart }, ImU32( UIColor::Arrow ) );

                    drawList->AddLine( { rightArrowX, yStart }, { rightArrowX, tipEnd }, ImU32( UIColor::Arrow ), 3.0f * cMenuScaling );
                    drawList->AddTriangleFilled( { rightArrowX - 10.0f, tipStart }, { rightArrowX, tipEnd }, { rightArrowX + 10.0f, tipStart }, ImU32( UIColor::Arrow ) );
                }
                else
                {

                }
            }

            if ( _activeCell == gridPos )
                drawList->AddRect( { topLeft.x + 24, topLeft.y + 24 }, { bottomRight.x - 24, bottomRight.y - 24 }, ImU32( UIColor::ActiveCellBorder ), 0, 0, 2.0f );

            ImGui::PopClipRect();
        }
    }

    topLeft.y = float( cGridTop + _viewportSize.height * cGridCellHeight );
    drawList->AddLine( { 0, topLeft.y - 1 }, { _size.x, topLeft.y - 1 }, ImU32( UIColor::TableBorders ) );

    ImGui::PopFont();

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


