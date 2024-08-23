#include "ImageReaderWindow.h"
#include "MainWindow.h"
#include "Serializer.h"
#include "FileDialog.h"
#include "ImGuiHelpers.h"
#include "./../Codecs/imagedecoder.h"
#include "imgui/imgui_internal.h"

#include <sstream>

ACMB_GUI_NAMESPACE_BEGIN

std::string GetFilters()
{
    const auto extensions = ImageDecoder::GetAllExtensions();
    std::ostringstream ss;
    ss << ".*";
    for ( const auto& extension : extensions )
    {
        ss << "," << extension;
    }
    return ss.str();
}

ImageReaderWindow::ImageReaderWindow( const Point& gridPos )
    : PipelineElementWindow( "Import Images", gridPos, PEFlags_NoInput | PEFlags_StrictlyOneOutput )
    , _workingDirectory( "." )
{
}

void ImageReaderWindow::DrawPipelineElementControls()
{
    const auto& style = ImGui::GetStyle();
    const float itemWidth = cElementWidth - 2.0f * style.WindowPadding.x;

    if ( ImGui::BeginListBox( "##ImageList", { itemWidth, 110 } ) )
    {
        for ( int i = 0; i < int( _fileNames.size() ); ++i )
        {
            const bool is_selected = (_selectedItemIdx == i);
            const std::string shortName = _fileNames[i].substr( _fileNames[i].find_last_of( "\\/" ) + 1 );
            if ( ImGui::Selectable( shortName.c_str(), is_selected ) )
            {
                _selectedItemIdx = i;
                ResetPreview();
            }
            // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
            if ( is_selected )
                ImGui::SetItemDefaultFocus();
        }
        ImGui::EndListBox();
    }

    ImGui::Text( "%d frames in %d files", int( _frameCount ), int( _fileNames.size() ) );

    auto fileDialog = FileDialog::Instance();
    const auto openDialogName = "SelectImagesDialog##" + _name;

    UI::Button( "Select Images", { itemWidth, 0 }, [&]
    {
        _showPreview = false;
        ImGui::CloseCurrentPopup();

        static auto filters = GetFilters();
        fileDialog.OpenDialog( openDialogName, "Select Images", filters.c_str(), _workingDirectory.c_str(), 0 );
        ResetProgress( PropagationDir::Forward );
    }, "Add images to the importing list", this );

    UI::Button( "Clear List", { itemWidth, 0 }, [&]
    {
        _showPreview = false;
        ImGui::CloseCurrentPopup();

        _fileNames.clear();
        _frameCount = 0;
        _selectedItemIdx = 0;
        _taskNumberToFileIndex.clear();
        ResetProgress( PropagationDir::Forward );
    }, "Delete all images from the importing list", this );

    //UI::Checkbox( "Invert Order", &_invertOrder, "Invert the order of the selected images" );

    if ( fileDialog.Display( openDialogName, {}, { 300 * cMenuScaling, 200 * cMenuScaling } ) )
    {
        _workingDirectory = fileDialog.GetCurrentPath() + "/";
        // action if OK
        if ( fileDialog.IsOk() )
        {
            const auto selection = fileDialog.GetSelection();
            for ( const auto& it : selection )
            {
                const auto path = _workingDirectory + it.first;

                try
                {
                    auto pDecoder = ImageDecoder::Create( path );
                    _fileNames.push_back( path );
                    _frameCount += pDecoder->GetFrameCount();
                    _taskNumberToFileIndex[int( _frameCount - 1 )] = int( _fileNames.size() - 1 );
                }
                catch ( std::exception& e )
                {
                    _error = e.what();
                    _showError = true;
                }
            }
        }

        // close
        fileDialog.Close();
    }
}

Expected<void, std::string> ImageReaderWindow::GeneratePreviewBitmap()
{
    if ( _fileNames.empty() )
        return unexpected( "No images in the list" );

    if ( _selectedItemIdx >= int( _fileNames.size() ) )
        return unexpected( "No image selected" );

    if ( _fileNames[_selectedItemIdx].empty() )
        return unexpected( "Selected file name is empty" );
    
    auto pDecoder = ImageDecoder::Create( _fileNames[_selectedItemIdx] );
    const auto mainWindow = ImGui::FindWindowByName( "acmb" );
    _pPreviewBitmap = pDecoder->ReadPreview( Size{ std::min( int( mainWindow->Size.x * 0.5f ), 1280 ),  std::min( int( mainWindow->Size.y * 0.5f ), 720 ) } );
    return {};
}

Expected<IBitmapPtr, std::string> ImageReaderWindow::RunTask( size_t i )
{
    try
    {
        const int idx = int( _invertOrder ? _fileNames.size() - 1 - i : i );
        const auto it = _taskNumberToFileIndex.lower_bound( idx );
        if ( it == _taskNumberToFileIndex.end() )
            return unexpected( "Incorrect index" );

        if ( !_pDecoder )
        {
            _pDecoder = ImageDecoder::Create( _fileNames[it->second] );
        }

        auto res = _pDecoder->ReadBitmap();
        if ( it->first == idx )
            _pDecoder.reset();

        return res;
    }
    catch ( std::exception& e )
    {
        return unexpected( e.what() );
    }
}

Expected<Size, std::string> ImageReaderWindow::GetBitmapSize()
{
    if ( _fileNames.empty() )
        return unexpected( "No images in the list" );

    if ( _selectedItemIdx >= int( _fileNames.size() ) )
        return unexpected( "No image selected" );

    if ( _fileNames[_selectedItemIdx].empty() )
        return unexpected( "Selected file name is empty" );
    
    try
    {
        auto pDecoder = ImageDecoder::Create( _fileNames[_selectedItemIdx] );
        const auto res = Size{ int( pDecoder->GetWidth() ), int( pDecoder->GetHeight() ) };
        pDecoder->Detach();
        return res;
    }
    catch ( std::exception& e )
    {
        return unexpected( e.what() );
    }
}

void ImageReaderWindow::Serialize( std::ostream& out ) const
{
    PipelineElementWindow::Serialize( out );
    gui::Serialize( _workingDirectory, out );
    gui::Serialize( _fileNames, out );
    gui::Serialize( _selectedItemIdx, out );
    gui::Serialize( _invertOrder, out );
}

bool ImageReaderWindow::Deserialize( std::istream& in )
{
    if ( !PipelineElementWindow::Deserialize( in ) ) return false;
    _workingDirectory = gui::Deserialize<std::string>( in, _remainingBytes );
    _fileNames = gui::Deserialize<std::vector<std::string>>( in, _remainingBytes );
    _selectedItemIdx = gui::Deserialize<int>( in, _remainingBytes );
    _invertOrder = gui::Deserialize<bool>( in, _remainingBytes );

    for ( size_t i = 0; i < _fileNames.size(); ++i )
    {
        auto& fileName = _fileNames[i];        
        _frameCount += ImageDecoder::Create( fileName )->GetFrameCount();
        _taskNumberToFileIndex[int( _frameCount - 1)] = int( i );
    }
    return true;
}

int ImageReaderWindow::GetSerializedStringSize() const
{
    return PipelineElementWindow::GetSerializedStringSize()
        + gui::GetSerializedStringSize( _workingDirectory )
        + gui::GetSerializedStringSize( _fileNames )
        + gui::GetSerializedStringSize( _selectedItemIdx )
        + gui::GetSerializedStringSize( _invertOrder );
}

std::string ImageReaderWindow::GetTaskName( size_t taskNumber ) const
{
    const auto it = _taskNumberToFileIndex.lower_bound( int( taskNumber ) );
    return ( it == _taskNumberToFileIndex.end() ) ? std::string{} : _fileNames[it->second];
}

size_t ImageReaderWindow::GetTaskCount()
{
    return _frameCount;
}

REGISTER_TOOLS_ITEM( ImageReaderWindow )

ACMB_GUI_NAMESPACE_END
