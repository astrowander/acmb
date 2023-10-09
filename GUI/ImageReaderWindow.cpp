#include "ImageReaderWindow.h"
#include "MainWindow.h"
#include "ImGuiFileDialog/ImGuiFileDialog.h"

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
: PipelineElementWindow( "Image Reader", gridPos, PEFlags_NoInput | PEFlags_StrictlyOneOutput )
, _workingDirectory( "." )
{
}

void ImageReaderWindow::DrawPipelineElementControls()
{
    const auto& style = ImGui::GetStyle();
    const float itemWidth = cElementWidth - 2.0f * style.WindowPadding.x;    

    if ( ImGui::BeginListBox( "##ImageList", { itemWidth, 85 * cMenuScaling } ) )
    {
        for ( int i = 0; i < _fileNames.size(); ++i )
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

    auto pFileDialog = ImGuiFileDialog::Instance();
    const auto openDialogName = "SelectImagesDialog##" + _name;

    if ( ImGui::Button( "Select Images", { itemWidth, 0 } ) )
    {
        static auto filters = GetFilters();
        pFileDialog->OpenDialog( openDialogName, "Select Images", filters.c_str(), _workingDirectory.c_str(), 0);
    }

    if ( pFileDialog->Display( openDialogName, {}, { 300 * cMenuScaling, 200 * cMenuScaling } ) )
    {
        // action if OK
        if ( pFileDialog->IsOk() )
        {
            _workingDirectory = pFileDialog->GetCurrentPath();

            const auto selection = pFileDialog->GetSelection();
            for ( const auto& it : selection )
                _fileNames.push_back( _workingDirectory + "/" + it.first);

            _taskCount = _fileNames.size();
        }

        // close
        ImGuiFileDialog::Instance()->Close();
    }
}

std::expected<IBitmapPtr, std::string> ImageReaderWindow::RunTask( size_t i )
{
    try
    {
        return IBitmap::Create( _fileNames[i] );
    }
    catch ( std::exception& e )
    {
        return std::unexpected( e.what() );
    }
}

REGISTER_TOOLS_ITEM( ImageReaderWindow )

ACMB_GUI_NAMESPACE_END