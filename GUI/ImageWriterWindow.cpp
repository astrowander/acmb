#include "ImageWriterWindow.h"
#include "MainWindow.h"
#include "FileDialog.h"
#include "Serializer.h"
#include "ImGuiHelpers.h"

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

static std::string GetFormatList()
{
    const auto extensions = ImageEncoder::GetAllExtensions();
    std::ostringstream ss;
    for ( const auto& extension : extensions )
    {
        ss << extension << '\0';
    }

    const auto res = ss.str();
    return { res.begin(), std::prev( res.end() ) };
}

ImageWriterWindow::ImageWriterWindow( const Point& gridPos )
: PipelineElementWindow( "Export Images", gridPos, PEFlags_StrictlyOneInput | PEFlags_NoOutput )
{
    _formatList = GetFormatList();
}

void ImageWriterWindow::DrawPipelineElementControls()
{
    ImGui::Checkbox( "Keep Original Name", &_keepOriginalFileName );
    ImGui::SetTooltipIfHovered( "If checked then the images will be saved with their original names. Otherwise the custom name will be used (with number postfixes for multiple images)", cMenuScaling );
    ImGui::Separator();

    const float itemWidth = 100.0f * cMenuScaling;
    auto fileDialog = FileDialog::Instance();

    if ( _keepOriginalFileName )
    {
        ImGui::Text( "Directory Path" );
        ImGui::SetNextItemWidth( itemWidth );
        ImGui::InputText( "##Directory", ( char* ) _workingDirectory.c_str(), 1024, ImGuiInputTextFlags_ReadOnly );
        if ( ImGui::Button( "Select Directory", { itemWidth, 0 } ) )
        {
            fileDialog.OpenDialog( "SelectOutputFile", "Select Directory", nullptr, _workingDirectory.c_str(), 0 );
        }

        ImGui::Combo( "Format", &_formatId, _formatList.c_str() );
    }
    else
    {
        ImGui::Text( "File Name" );
        ImGui::SetNextItemWidth( itemWidth );
        ImGui::InputText( "##File Name", ( char* ) _fileName.c_str(), 1024, ImGuiInputTextFlags_ReadOnly );

        if ( ImGui::Button( "Select File", { itemWidth, 0 } ) )
        {
            static auto filters = GetFilters();
            fileDialog.OpenDialog( "SelectOutputFile", "Select File", filters.c_str(), _workingDirectory.c_str(), 0 );
        }
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

void ImageWriterWindow::Serialize( std::ostream& out ) const
{
    PipelineElementWindow::Serialize( out );
    gui::Serialize( _workingDirectory, out );
    gui::Serialize( _fileName, out );
    gui::Serialize( _formatId, out );
    gui::Serialize( _keepOriginalFileName, out );
}

void ImageWriterWindow::Deserialize( std::istream& in )
{
    PipelineElementWindow::Deserialize( in );
    _workingDirectory = gui::Deserialize<std::string>( in, _remainingBytes );
    _fileName = gui::Deserialize<std::string>( in, _remainingBytes );
    _formatId = gui::Deserialize<int>( in, _remainingBytes );
    _keepOriginalFileName = gui::Deserialize<bool>( in, _remainingBytes );
}

int ImageWriterWindow::GetSerializedStringSize() const
{
    return PipelineElementWindow::GetSerializedStringSize() 
        + gui::GetSerializedStringSize( _workingDirectory )
        + gui::GetSerializedStringSize( _fileName )
        + gui::GetSerializedStringSize( _formatId )
        + gui::GetSerializedStringSize( _keepOriginalFileName );
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
    std::string finalName;
    if ( _keepOriginalFileName )
    {
        auto taskName = GetTaskName( taskNumber );
        const size_t lastSlashPos = taskName.find_last_of( "\\/" );
        const size_t lastDotPos = taskName.find_last_of( '.' );

        static const auto extensions = ImageEncoder::GetAllExtensions();
        auto extensionIt = extensions.begin();
        std::advance( extensionIt, _formatId );

        finalName = _workingDirectory + "/" + taskName.substr( lastSlashPos + 1, lastDotPos - lastSlashPos - 1 ) + *extensionIt;
    }
    else
    {
        const size_t dotPos = _fileName.find_last_of( '.' );
        finalName = (taskNumber == 0) ? _fileName : (_fileName.substr( 0, dotPos ) + "_" + std::to_string( taskNumber ) + _fileName.substr( dotPos ) );
    }

    IBitmap::Save( pSource, finalName );
    return nullptr;
}

std::vector<std::string> ImageWriterWindow::RunAllTasks()
{
    auto pPrimaryInput = GetPrimaryInput();
    if ( !pPrimaryInput )
        return { "No primary input for the'" + _name + "' element" };

    _taskCount = pPrimaryInput->GetTaskCount();
    if ( _taskCount == 0 )
        return { "No input frames for the'" + _name + "' element" };

    if ( _keepOriginalFileName && _workingDirectory.empty() )
        return { "Working directory for the'" + _name + "' element is not specified" };

    if ( !_keepOriginalFileName && _fileName.empty() )
        return { "File name for the'" + _name + "' element is not specified" };

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