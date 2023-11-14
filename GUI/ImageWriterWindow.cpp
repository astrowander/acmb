#include "ImageWriterWindow.h"
#include "MainWindow.h"
#include "FileDialog.h"
#include "Serializer.h"

#include "./../Core/bitmap.h"
#include "./../Codecs/imageencoder.h"
#include <sstream>

ACMB_GUI_NAMESPACE_BEGIN

static std::string GetFilters()
{
    const auto extensions = ImageEncoder::GetAllExtensions();
    std::ostringstream ss;
    ss << ".*";
    for ( const auto& extension : extensions )
    {
        ss << "," << extension;
    }
    return ss.str();
}

ImageWriterWindow::ImageWriterWindow( const Point& gridPos )
: PipelineElementWindow( "Image Writer", gridPos, PEFlags_StrictlyOneInput | PEFlags_NoOutput )
{
}

void ImageWriterWindow::DrawPipelineElementControls()
{
    ImGui::Text( "Choose output file format:", "" );
    ImGui::InputText( "File Name", ( char* ) _fileName.c_str(), 1024, ImGuiInputTextFlags_ReadOnly );

    auto fileDialog = FileDialog::Instance();
    if ( ImGui::Button( "Select File" ) )
    {
        static auto filters = GetFilters();
        fileDialog.OpenDialog( "SelectOutputFile", "Select File", filters.c_str(), _workingDirectory.c_str(), 0 );
    }

    if ( fileDialog.Display( "SelectOutputFile", {}, { 300 * cMenuScaling, 200 * cMenuScaling } ) )
    {
        // action if OK
        if ( fileDialog.IsOk() )
        {
            _workingDirectory = fileDialog.GetCurrentPath();
            _fileName = fileDialog.GetFilePathName();
        }

        // close
        fileDialog.Close();
    }
}

void ImageWriterWindow::Serialize( std::ostream& out )
{
    PipelineElementWindow::Serialize( out );
    acmb::gui::Serialize( _workingDirectory, out );
    acmb::gui::Serialize( _fileName, out );
}

void ImageWriterWindow::Deserialize( std::istream& in )
{
    PipelineElementWindow::Deserialize( in );
    _workingDirectory = acmb::gui::Deserialize<std::string>( in );
    _fileName = acmb::gui::Deserialize<std::string>( in );
}

/*std::expected<IBitmapPtr, std::string> ImageWriterWindow::RunTask(size_t i)
{
    auto pInput = GetLeftInput();
    if ( !pInput )
        pInput = GetTopInput();

    if ( !pInput )
        return std::unexpected( "No input element" );
   
    const auto taskRes = pInput->RunTaskAndReportProgress( i );
    if ( !taskRes.has_value() )
        return std::unexpected( taskRes.error() );

    const size_t dotPos = _fileName.find_last_of('.');
    IBitmap::Save(taskRes.value(), (i == 0) ? _fileName : ( _fileName.substr(0, dotPos) + "_" + std::to_string(i) + _fileName.substr( dotPos ) ) );
    return nullptr;
}*/

IBitmapPtr ImageWriterWindow::ProcessBitmapFromPrimaryInput( IBitmapPtr pSource, size_t taskNumber )
{
    const size_t dotPos = _fileName.find_last_of( '.' );
    IBitmap::Save( pSource, (taskNumber == 0) ? _fileName : (_fileName.substr( 0, dotPos ) + "_" + std::to_string( taskNumber ) + _fileName.substr( dotPos )) );
    return nullptr;
}

std::vector<std::string> ImageWriterWindow::RunAllTasks()
{
    auto pPrimaryInput = GetPrimaryInput();
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