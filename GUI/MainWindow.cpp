#include "MainWindow.h"
#include "ImageWriterWindow.h"
#include "FontRegistry.h"
#include "FileDialog.h"
#include "ImGuiHelpers.h"
#include "./../Tools/SystemTools.h"
#include "./../Cuda/CudaInfo.h"
#include <fstream>
#include <thread>

#ifdef _WIN32
#include <atltypes.h>
#include <WinUser.h>
#elif defined ( __linux__ )
#include <GLFW/glfw3.h>
#endif

ACMB_GUI_NAMESPACE_BEGIN

static constexpr float cMenuButtonSize = 50.0f;

static constexpr float cHeadRowHeight = 25;
static constexpr float cGridTop = cMenuButtonSize + 105.0f;

static constexpr float cGridLeft = 30;
static constexpr float cGridCellPadding = 25.0f;

static constexpr float cGridCellWidth = PipelineElementWindow::cElementWidth + 2.0f * cGridCellPadding;
static constexpr float cGridCellHeight = PipelineElementWindow::cElementHeight + 2.0f * cGridCellPadding;

MainWindow::MainWindow( const ImVec2& pos, const ImVec2& size, const FontRegistry& fontRegistry )
    : Window( "acmb", size )
    , _fontRegistry( fontRegistry )
{
    SetPos( pos );
    _viewportSize = { (  int( size.x ) - int( cGridLeft ) ) / int( cGridCellWidth ), ( int( size.y ) - int(cGridTop)) / int(cGridCellHeight)};

    MenuItemsHolder::GetInstance().AddItem( "Run", 1, "\xef\x81\x8B", "Run", "Start processing", [this] (Point)
    {
        _errors.clear();
        _isBusy = true;
        LockInterface();

        std::thread process( [&]
        {
            _startTime = std::chrono::high_resolution_clock::now();
            _durationString.clear();

            if ( _writers.empty() )
                _errors.emplace_back( "There are not 'Export' tools in the scheme" );

            for ( auto pWriter : _writers )
            {
                pWriter.second.lock()->ResetTasks();
                const auto errors = pWriter.second.lock()->ExportAllImages();
                _errors.insert( _errors.end(), errors.begin(), errors.end() );
            }

            _isBusy = false;
            _showResultsPopup = true;
            UnlockInterface();
        } );
        process.detach();
    } );

    const auto acmbPath = GetAcmbPath();

    MenuItemsHolder::GetInstance().AddItem( "Project", 2, "\xef\x83\x87", "Save", "Write the project to an .acmb file", [acmbPath] (Point)
    {
        FileDialog::Instance().OpenDialog( "SaveProjectDialog", "Save Table", ".acmb", ( acmbPath + "/GUI/presets/" ).c_str(), 1 );
    } );

    MenuItemsHolder::GetInstance().AddItem( "Project", 1, "\xef\x81\xbc", "Open", "Read the project from an .acmb file", [acmbPath] ( Point )
    {
        FileDialog::Instance().OpenDialog( "OpenProjectDialog", "Load Table", ".acmb", ( acmbPath + "/GUI/presets/" ).c_str(), 1 );
    } );

    MenuItemsHolder::GetInstance().AddItem( "Help", 1, "\xef\x84\xa8", "Help", "Show modal window with instructions", [this] ( Point )
    {
        _showHelpPopup = true;
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
    const bool isMouseDoubleClicked = ImGui::IsMouseDoubleClicked( ImGuiMouseButton_Left );
    const bool isMouseClicked = ImGui::IsMouseClicked( ImGuiMouseButton_Left );

    if ( !isMouseClicked && !isMouseDoubleClicked )
        return;

    auto mousePos = ImGui::GetMousePos();
    mousePos.x -= cGridLeft;
    mousePos.y -= cGridTop;

    if ( mousePos.x < 0 || mousePos.y < 0 )
        return;

    int col = int( mousePos.x ) / int( cGridCellWidth ) + _viewportStart.x;
    int row = int( mousePos.y ) / int( cGridCellHeight ) + _viewportStart.y;

    const auto gridIdx = row * cGridSize.width + col;
    std::shared_ptr<PipelineElementWindow> pElement;
    if (gridIdx >=0 && gridIdx < cGridSize.width * cGridSize.height )
        pElement = _grid[gridIdx];
    
    mousePos.x -= ( col - _viewportStart.x ) * cGridCellWidth;
    mousePos.y -= ( row - _viewportStart.y ) * cGridCellHeight;

    if ( isMouseDoubleClicked  )
    { 
        if ( !pElement )
            return;

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
    else if ( isMouseClicked )
    {
        if ( col < 0 || col >= cGridSize.width || row < 0 || row >= cGridSize.height )
            return;

        if ( mousePos.x < cGridCellPadding || mousePos.y < cGridCellPadding || mousePos.x > cGridCellWidth - cGridCellPadding || mousePos.y > cGridCellHeight - cGridCellPadding )
            return;

        _activeCell = { .x = col, .y = row };
    }
}

void MainWindow::OpenProject()
{
    for ( auto& pElement : _grid )
        pElement.reset();

    auto reportError = [this] ( const std::string msg )
    {
        _errors.push_back( msg );
        _showResultsPopup = true;
        return FileDialog::Instance().Close();
    };
    std::string filePath = FileDialog::Instance().GetFilePathName();
    std::ifstream fin( filePath, std::ios_base::in | std::ios_base::binary );

    _errors.clear();
    if ( !fin )
        return reportError( "Unable to open file" );

    fin.seekg( 0, std::ios_base::end );
    int streamSize = int( fin.tellg() );
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
        return reportError( "Schema is too large" );

    int charCount = actualGridSize.width * actualGridSize.height;
    if ( streamSize < charCount + 6 )
        return reportError( "File is corrupted" );

    std::string serialized( charCount, '\0' );
    fin.read( serialized.data(), charCount );

    for ( int i = 0; i < actualGridSize.height; ++i )
    for ( int j = 0; j < actualGridSize.width; ++j )
    {
        if ( uint8_t menuOrder = serialized[i * actualGridSize.width + j] )
        {
            MenuItemsHolder::GetInstance().GetItems().at( "Tools" ).at( menuOrder )->action( Point{ .x = int( j ), .y = int( i ) } );
            if ( streamSize > charCount + 6 )
            {
                _grid[i * cGridSize.height + j]->Deserialize( fin );
                const auto serializedInputs = _grid[i * cGridSize.height + j]->GetActualInputs();

                if ( serializedInputs.left != PipelineElementWindow::RelationType::None )
                {
                    _grid[i * cGridSize.height + j]->SetLeftInput( _grid[i * cGridSize.height + j - 1] );
                    _grid[i * cGridSize.height + j]->SetLeftRelationType( serializedInputs.left );
                    _grid[i * cGridSize.height + j - 1]->SetRightOutput( _grid[i * cGridSize.height + j] );
                    _grid[i * cGridSize.height + j - 1]->SetRightRelationType( serializedInputs.left );
                }
                else if ( j > 0 )
                {
                    _grid[i * cGridSize.height + j]->SetLeftInput( nullptr );
                    if ( _grid[i * cGridSize.height + j - 1] )
                        _grid[i * cGridSize.height + j - 1]->SetRightOutput( nullptr );
                }

                if ( serializedInputs.top != PipelineElementWindow::RelationType::None )
                {
                    _grid[i * cGridSize.height + j]->SetTopInput( _grid[( i - 1 ) * cGridSize.height + j] );
                    _grid[i * cGridSize.height + j]->SetTopRelationType( serializedInputs.top );
                    _grid[( i - 1 ) * cGridSize.height + j]->SetBottomOutput( _grid[i * cGridSize.height + j] );
                    _grid[(i - 1) * cGridSize.height + j]->SetBottomRelationType( serializedInputs.top );
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
        _showResultsPopup = true;
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

    for ( int i = 0; i < actualGridSize.height; ++i )
        for ( int j = 0; j < actualGridSize.width; ++j )
        {
            chars[i * actualGridSize.width + j] = ( ( _grid[i * cGridSize.height + j] ) ? _grid[i * cGridSize.height + j]->GetMenuOrder() : 0 );
        }

    fout.write( chars.data(), chars.size() );

    for ( int i = 0; i < actualGridSize.height; ++i )
        for ( int j = 0; j < actualGridSize.width; ++j )
        {
            if ( _grid[i * cGridSize.height + j] )
                _grid[i * cGridSize.height + j]->Serialize( fout );
        }
}

void MainWindow::DrawMenu()
{
    for ( const auto& it : MenuItemsHolder::GetInstance().GetItems() )
    {
        const auto& category = it.first;
        const auto& items = it.second;

        const auto itemCount = items.size();
        const float menuWidth = itemCount * cMenuButtonSize + ( itemCount - 1 ) * ImGui::GetStyle().ItemSpacing.x;

        ImGui::BeginChild( category.c_str(), { menuWidth, 0 } );

        ImGui::PushFont( _fontRegistry.bold );
        ImGui::SeparatorText( category.c_str() );
        ImGui::PopFont();

        for ( auto& item : items )
        {
            ImGui::PushFont( _fontRegistry.icons );

            const auto oldPos = ImGui::GetCursorPos();
            UI::Button( item.second->icon, { cMenuButtonSize, cMenuButtonSize }, [&]
            {
                item.second->action( _activeCell );
            }, item.second->tooltip );

            ImGui::PopFont();

            const float textWidth = ImGui::CalcTextSize( item.second->caption.c_str() ).x;
            ImGui::SetCursorPos( { oldPos.x + ( cMenuButtonSize - textWidth ) * 0.5f, oldPos.y + cMenuButtonSize + ImGui::GetStyle().ItemSpacing.y } );
            ImGui::Text( "%s", item.second->caption.c_str() );
            ImGui::SetCursorPos( { oldPos.x + cMenuButtonSize + ImGui::GetStyle().ItemSpacing.x, oldPos.y});
        }

        ImGui::EndChild();

        ImGui::SameLine();
        ImGui::Dummy( { 2.0f * ImGui::GetStyle().ItemSpacing.x, 0 } );
        ImGui::SameLine();
    }

    const auto cachedPos = ImGui::GetCursorPos();

    if ( _fontRegistry.bold )
        ImGui::PushFont( _fontRegistry.bold );

    ImGui::SeparatorText( "General Settings" );

    if ( _fontRegistry.bold )
        ImGui::PopFont();

    ImGui::SetCursorPos( { cachedPos.x, cachedPos.y + ImGui::GetTextLineHeight() + ImGui::GetStyle().ItemSpacing.y } );
    if ( cuda::isCudaAvailable() )
        UI::Checkbox( "Enable CUDA", &_enableCuda, "Performs computations on a graphic card if available" );

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

void MainWindow::DrawDialog()
{
    if ( !_lockInterface && !_showHelpPopup )
    {
        ProcessKeyboardEvents();
        ProcessMouseEvents();
    }

    DrawMenu();

    ImGui::PushFont( _fontRegistry.byDefault );

    ImGui::SetCursorPos( { 0, cGridTop - cHeadRowHeight } );
    UI::Button( "##ClearTable", { cGridLeft, cHeadRowHeight }, [&]
    {
        for ( auto& pElement : _grid )
            pElement.reset();
    }, "Clear table" );

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
                    const float lineEnd = tipEnd - 3.0f * cMenuScaling;

                    drawList->AddLine( { topLeft.x, topArrowY }, { lineEnd, topArrowY }, ImU32( UIColor::Arrow ), 3.0f * cMenuScaling );
                    drawList->AddTriangleFilled( { tipStart, topArrowY - 10.0f }, { tipEnd, topArrowY }, { tipStart, topArrowY + 10.0f }, ImU32( UIColor::Arrow ) );

                    drawList->AddLine( { topLeft.x, centerArrowY }, { lineEnd, centerArrowY }, ImU32( UIColor::Arrow ), 3.0f * cMenuScaling );
                    drawList->AddTriangleFilled( { tipStart, centerArrowY - 10.0f }, { tipEnd, centerArrowY }, { tipStart, centerArrowY + 10.0f }, ImU32( UIColor::Arrow ) );

                    drawList->AddLine( { topLeft.x, bottomArrowY }, { lineEnd, bottomArrowY }, ImU32( UIColor::Arrow ), 3.0f * cMenuScaling );
                    drawList->AddTriangleFilled( { tipStart, bottomArrowY - 10.0f }, { tipEnd, bottomArrowY }, { tipStart, bottomArrowY + 10.0f }, ImU32( UIColor::Arrow ) );
                }
                else
                {
                    const float startX = topLeft.x - 2.0f * cMenuScaling;

                    drawList->AddLine( { startX, topArrowY - 2.0f * cMenuScaling }, { tipEnd, centerArrowY }, ImU32( UIColor::Arrow ), 2.0f * cMenuScaling );
                    drawList->AddLine( { startX, centerArrowY }, { tipEnd, centerArrowY }, ImU32( UIColor::Arrow ), 3.0f * cMenuScaling );
                    drawList->AddLine( { startX, bottomArrowY + 2.0f * cMenuScaling }, { tipEnd, centerArrowY }, ImU32( UIColor::Arrow ), 2.0f * cMenuScaling );
                }
            }

            const float leftArrowX = topLeft.x + cGridCellWidth * 0.5f - 2.0f * cGridCellPadding;
            const float centerArrowX = topLeft.x + cGridCellWidth * 0.5f;
            const float rightArrowX = topLeft.x + cGridCellWidth * 0.5f + 2.0f * cGridCellPadding;

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
                    const float yStart = topLeft.y - 1.0f * cMenuScaling;
                    const float lineEnd = tipEnd - 3.0f * cMenuScaling;

                    drawList->AddLine( { leftArrowX, yStart }, { leftArrowX, lineEnd }, ImU32( UIColor::Arrow ), 3.0f * cMenuScaling );
                    drawList->AddTriangleFilled( { leftArrowX - 10.0f, tipStart }, { leftArrowX, tipEnd }, { leftArrowX + 10.0f, tipStart }, ImU32( UIColor::Arrow ) );

                    drawList->AddLine( { centerArrowX, yStart }, { centerArrowX, lineEnd }, ImU32( UIColor::Arrow ), 3.0f * cMenuScaling );
                    drawList->AddTriangleFilled( { centerArrowX - 10.0f, tipStart }, { centerArrowX, tipEnd }, { centerArrowX + 10.0f, tipStart }, ImU32( UIColor::Arrow ) );

                    drawList->AddLine( { rightArrowX, yStart }, { rightArrowX, lineEnd }, ImU32( UIColor::Arrow ), 3.0f * cMenuScaling );
                    drawList->AddTriangleFilled( { rightArrowX - 10.0f, tipStart }, { rightArrowX, tipEnd }, { rightArrowX + 10.0f, tipStart }, ImU32( UIColor::Arrow ) );
                }
                else
                {
                    const float yStart = topLeft.y - 2.0f * cMenuScaling;
                    drawList->AddLine( { leftArrowX - 2.0f * cMenuScaling, yStart }, { centerArrowX, tipEnd }, ImU32( UIColor::Arrow ), 2.0f * cMenuScaling );
                    drawList->AddLine( { centerArrowX, yStart }, { centerArrowX, tipEnd }, ImU32( UIColor::Arrow ), 3.0f * cMenuScaling );
                    drawList->AddLine( { rightArrowX + 2.0f * cMenuScaling, yStart }, { centerArrowX, tipEnd }, ImU32( UIColor::Arrow ), 2.0f * cMenuScaling );
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

    if ( _showResultsPopup )
    {
        if ( _durationString.empty() )
        {
            const auto ms = std::chrono::duration_cast< std::chrono::milliseconds >(std::chrono::high_resolution_clock::now() - _startTime).count();
            _durationString = "Elapsed " + std::to_string( ms / 1000 ) + "s " + std::to_string( ms % 1000 ) + "ms";
        }

        if ( _errors.empty() )
            return;

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

void MainWindow::Show()
{
    Window::Show();

    for ( int i = 0; i < int( _grid.size() ); ++i )
    {
        const int x = i % cGridSize.width;
        const int y = i / cGridSize.width;

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
