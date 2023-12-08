#include "ImageReaderWindow.h"
#include "MainWindow.h"
#include "Serializer.h"
#include "FileDialog.h"
#include "ImGuiHelpers.h"

#include "./../Codecs/imagedecoder.h"

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

    if ( ImGui::BeginListBox( "##ImageList", { itemWidth, 128 } ) )
    {
        for ( int i = 0; i < int( _fileNames.size() ); ++i )
        {
            const bool is_selected = ( _selectedItemIdx == i );
            const std::string shortName = _fileNames[i].substr( _fileNames[i].find_last_of( "\\/" ) + 1 );
            if ( ImGui::Selectable( shortName.c_str(), is_selected ) )
                _selectedItemIdx = i;

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
        static auto filters = GetFilters();
        fileDialog.OpenDialog( openDialogName, "Select Images", filters.c_str(), _workingDirectory.c_str(), 0 );
        _completedTaskCount = 0;
    }, "Add images to the importing list" );

    UI::Button( "Clear List", { itemWidth, 0 }, [&]
    {
        _fileNames.clear();
        _selectedItemIdx = 0;
        _completedTaskCount = 0;
    }, "Delete all images from the importing list" );

    if ( fileDialog.Display( openDialogName, {}, { 300 * cMenuScaling, 200 * cMenuScaling } ) )
    {
        // action if OK
        if ( fileDialog.IsOk() )
        {
            _workingDirectory = fileDialog.GetCurrentPath();

            const auto selection = fileDialog.GetSelection();
            for ( const auto& it : selection )
                _fileNames.push_back( _workingDirectory + "/" + it.first);

            _taskCount = _fileNames.size();
        }

        // close
        fileDialog.Close();
    }
}

Expected<IBitmapPtr, std::string> ImageReaderWindow::RunTask( size_t i )
{
    try
    {
        return IBitmap::Create( _fileNames[i] );
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
    gui::Serialize( std::move( _fileNames ), out);
    gui::Serialize(_selectedItemIdx, out);    
}

void ImageReaderWindow::Deserialize(std::istream& in)
{
    PipelineElementWindow::Deserialize(in);
    _workingDirectory = gui::Deserialize<std::string>(in, _remainingBytes);
    _fileNames = gui::Deserialize<std::vector<std::string>>(in, _remainingBytes);
    _selectedItemIdx = gui::Deserialize<int>(in, _remainingBytes);

    _taskCount = _fileNames.size();
}

int ImageReaderWindow::GetSerializedStringSize() const
{
    return PipelineElementWindow::GetSerializedStringSize() 
        + gui::GetSerializedStringSize( _workingDirectory )
        + gui::GetSerializedStringSize( _fileNames )
        + gui::GetSerializedStringSize( _selectedItemIdx );
}

std::string ImageReaderWindow::GetTaskName( size_t taskNumber ) const
{
    return (taskNumber < _fileNames.size()) ? _fileNames[taskNumber] : std::string{};
}

REGISTER_TOOLS_ITEM( ImageReaderWindow )

ACMB_GUI_NAMESPACE_END
