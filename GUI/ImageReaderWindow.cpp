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

    if ( ImGui::BeginListBox( "##ImageList", { itemWidth, 90 } ) )
    {
        for ( int i = 0; i < int( _fileNames.size() ); ++i )
        {
            const bool is_selected = ( _selectedItemIdx == i );
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
        _selectedItemIdx = 0;
        ResetProgress( PropagationDir::Forward );
    }, "Delete all images from the importing list", this );

    UI::Checkbox( "Invert Order", &_invertOrder, "Invert the order of the selected images" );

    if ( fileDialog.Display( openDialogName, {}, { 300 * cMenuScaling, 200 * cMenuScaling } ) )
    {
        // action if OK
        if ( fileDialog.IsOk() )
        {
            _workingDirectory = fileDialog.GetCurrentPath();

            const auto selection = fileDialog.GetSelection();
            for ( const auto& it : selection )
                _fileNames.push_back( _workingDirectory + "/" + it.first);
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
        const size_t idx = _invertOrder ? _fileNames.size() - 1 - i : i;
        return IBitmap::Create( _fileNames[idx] );
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

void ImageReaderWindow::Serialize(std::ostream& out) const
{
    PipelineElementWindow::Serialize(out);
    gui::Serialize( _workingDirectory, out);
    gui::Serialize( _fileNames, out);
    gui::Serialize(_selectedItemIdx, out);    
    gui::Serialize( _invertOrder, out );
}

void ImageReaderWindow::Deserialize(std::istream& in)
{
    PipelineElementWindow::Deserialize(in);
    _workingDirectory = gui::Deserialize<std::string>(in, _remainingBytes);
    _fileNames = gui::Deserialize<std::vector<std::string>>(in, _remainingBytes);
    _selectedItemIdx = gui::Deserialize<int>(in, _remainingBytes);
    _invertOrder = gui::Deserialize<bool>( in, _remainingBytes );
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
    return (taskNumber < _fileNames.size()) ? _fileNames[taskNumber] : std::string{};
}

size_t ImageReaderWindow::GetTaskCount()
{
    return _fileNames.size();
}

REGISTER_TOOLS_ITEM( ImageReaderWindow )

ACMB_GUI_NAMESPACE_END
