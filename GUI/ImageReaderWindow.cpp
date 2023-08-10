#include "ImageReaderWindow.h"
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
: PipelineElementWindow( "Image Reader", gridPos, PipelineElementWindow::RequiredInOutFlags::NoInput | PipelineElementWindow::RequiredInOutFlags::StrictlyOneOutput )
, _workingDirectory( "." )
{
}

void ImageReaderWindow::DrawPipelineElementControls()
{
    ImGui::Text( "%s", "Image List" );

    const auto& style = ImGui::GetStyle();
    const float itemWidth = cElementWidth - 2.0f * style.FramePadding.x;    

    if ( ImGui::BeginListBox( "##ImageList", { itemWidth, 0 } ) )
    {
        for ( int i = 0; i < _fileNames.size(); ++i )
        {
            const bool is_selected = ( _selectedItemIdx == i );
            if ( ImGui::Selectable( _fileNames[i].c_str(), is_selected ) )
                _selectedItemIdx = i;

            // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
            if ( is_selected )
                ImGui::SetItemDefaultFocus();
        }
        ImGui::EndListBox();
    }

    auto pFileDialog = ImGuiFileDialog::Instance();

    if ( ImGui::Button( "Select Images", { itemWidth, 0 } ) )
    {
        static auto filters = GetFilters();
        pFileDialog->OpenDialog( "SelectImagesDialog", "Select Images", filters.c_str(), _workingDirectory.c_str(), 0);
    }

    if ( pFileDialog->Display( "SelectImagesDialog" ) )
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

ACMB_GUI_NAMESPACE_END