#include "ImageWriterWindow.h"
#include "MainWindow.h"
#include "ImGuiFileDialog/ImGuiFileDialog.h"

#include "./../Core/bitmap.h"
#include <sstream>

ACMB_GUI_NAMESPACE_BEGIN

ImageWriterWindow::ImageWriterWindow( const Point& gridPos )
: PipelineElementWindow( "Image Writer", gridPos, PEFlags_StrictlyOneInput | PEFlags_NoOutput )
{
}

void ImageWriterWindow::DrawPipelineElementControls()
{
    ImGui::Text( "Choose output file format:", "" );
    ImGui::RadioButton( ".ppm", ( int* ) &_fileFormat, int( FileFormat::Ppm ) );
    ImGui::RadioButton( ".tif", ( int* ) &_fileFormat, int( FileFormat::Tiff ) );
    ImGui::RadioButton( ".jpg", ( int* ) &_fileFormat, int( FileFormat::Jpeg ) );

    ImGui::InputText( "Working Directory", ( char* ) _workingDirectory.c_str(), 1024, ImGuiInputTextFlags_ReadOnly );
    
    static char buf[1024] = {};
    if ( ImGui::InputText( "Output File Name", buf, 1024 ) )
    {
        const size_t length = strlen( buf );
        _fileName = std::string( buf );
    }

    auto pFileDialog = ImGuiFileDialog::Instance();
    if ( ImGui::Button( "Select Directory" ) )
    {
        pFileDialog->OpenDialog( "SelectOutputDirectory", "Select Directory", nullptr, _workingDirectory.c_str(), 0);
    }

    if ( pFileDialog->Display( "SelectOutputDirectory", {}, { 300 * cMenuScaling, 200 * cMenuScaling } ) )
    {
        // action if OK
        if ( pFileDialog->IsOk() )
        {
            _workingDirectory = pFileDialog->GetCurrentPath();
        }

        // close
        ImGuiFileDialog::Instance()->Close();
    }
}

std::expected<IBitmapPtr, std::string> ImageWriterWindow::RunTask( size_t i )
{
    auto pPrimaryInput = GetLeftInput();
    if ( !pPrimaryInput )
        return std::unexpected( "No input element" );

   
    const auto taskRes = pPrimaryInput->RunTaskAndReportProgress( i );
    if ( !taskRes.has_value() )
        return std::unexpected( taskRes.error() );

    const std::string extension = ( _fileFormat == FileFormat::Ppm ) ? ".ppm" :
                                    ( _fileFormat == FileFormat::Tiff ) ? ".tif" : ".jpg";

    std::ostringstream ss;
    ss << _workingDirectory << "/" << _fileName << "_" << i << extension;
    IBitmap::Save( taskRes.value(), ss.str() );
    return nullptr;
    
}

std::vector<std::string> ImageWriterWindow::RunAllTasks()
{
    auto pPrimaryInput = GetLeftInput();
    if ( !pPrimaryInput )
        return { "No input element" };    

    _taskCount = pPrimaryInput->GetTaskCount();
    if ( _taskCount == 0 )
        return { "No tasks" };

    if ( _workingDirectory.empty() )
        return { "Working directory is not specified" };

    if ( _fileName.empty() )
        return { "File name is not specified" };

    std::vector<std::string> res;
    for ( size_t i = 0; i < _taskCount; ++i )
    {
        const auto taskRes = RunTaskAndReportProgress( i );
        if ( !taskRes.has_value() )
            res.push_back( taskRes.error() );
    }
    
    return res;
}

REGISTER_TOOLS_ITEM( ImageWriterWindow )

ACMB_GUI_NAMESPACE_END