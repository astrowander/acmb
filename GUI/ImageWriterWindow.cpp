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
    ImGui::InputText( "File Name", ( char* ) _fileName.c_str(), 1024, ImGuiInputTextFlags_ReadOnly );

    auto pFileDialog = ImGuiFileDialog::Instance();
    if ( ImGui::Button( "Select File" ) )
    {
        pFileDialog->OpenDialog( "SelectOutputFile", "Select File", nullptr, _workingDirectory.c_str(), 0);
    }

    if ( pFileDialog->Display( "SelectOutputFile", {}, { 300 * cMenuScaling, 200 * cMenuScaling } ) )
    {
        // action if OK
        if ( pFileDialog->IsOk() )
        {
            _workingDirectory = pFileDialog->GetCurrentPath();
            _fileName = pFileDialog->GetFilePathName();
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

    const size_t dotPos = _fileName.find_last_of('.');
    IBitmap::Save(taskRes.value(), (i == 0) ? _fileName : ( _fileName.substr(0, dotPos) + "_" + std::to_string(i) + _fileName.substr( dotPos ) ) );
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