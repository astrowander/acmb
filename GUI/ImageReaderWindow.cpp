#include "ImageReaderWindow.h"
#include "MainWindow.h"
#include "Serializer.h"
#include "ImGuiHelpers.h"
#include "./../Codecs/imagedecoder.h"
#include "imgui/imgui_internal.h"

#include <sstream>
#include "DarkFrameWindow.h"

ACMB_GUI_NAMESPACE_BEGIN



ImageReaderWindow::ImageReaderWindow( const Point& gridPos )
    : PipelineElementWindow( "Import Images", gridPos, PEFlags_NoInput | PEFlags_StrictlyOneOutput )
    , FileListUser(this)
{
}

void ImageReaderWindow::DrawPipelineElementControls()
{
    FileListUser::DrawControls();
}

void ImageReaderWindow::ResetTasks()
{
    PipelineElementWindow::ResetTasks();
    CleanUp();
}

Expected<void, std::string> ImageReaderWindow::GeneratePreviewBitmap()
{
    const auto mainWindow = ImGui::FindWindowByName( "acmb" );
    const Size size{ std::min( int( mainWindow->Size.x * 0.5f ), 1280 ),  std::min( int( mainWindow->Size.y * 0.5f ), 720 ) };
    auto resOrErr = ReadFramePreview( _previewedFrameNumber, size );
    if ( resOrErr )
        _pPreviewBitmap = *resOrErr;

    return unexpected( resOrErr.error() );
}

Expected<IBitmapPtr, std::string> ImageReaderWindow::RunTask( size_t i )
{
    try
    {
        return ReadFrame( int( i ) );
    }
    catch ( std::exception& e )
    {
        return unexpected( e.what() );
    }
}

Expected<Size, std::string> ImageReaderWindow::GetBitmapSize()
{
    return GetFrameSize( _previewedFrameNumber );
}

void ImageReaderWindow::OnSelectedFrameChanged(int idx)
{
    OnPreviewedFrameNumberChanged(idx);
}

void ImageReaderWindow::OnFileListChanged()
{
    FileListUser::OnFileListChanged();

    _showPreview = false;
    ImGui::CloseCurrentPopup();
    ResetProgress(PropagationDir::Forward);
}

std::string ImageReaderWindow::GetWindowName() const
{
    return _name;
}

std::string ImageReaderWindow::GetFileFilters() const
{
    return ImageDecoder::GetFilters();
}

void ImageReaderWindow::Serialize( std::ostream& out ) const
{
    PipelineElementWindow::Serialize( out );
    FileListUser::Serialize( out );
}

bool ImageReaderWindow::Deserialize( std::istream& in )
{
    if ( !PipelineElementWindow::Deserialize( in ) ) return false;

    if ( !FileListUser::Deserialize( in, _remainingBytes ) ) return false;

    /*for ( size_t i = 0; i < _fileNames.size(); ++i )
    {
        auto& fileName = _fileNames[i];
        try
        {
            auto pDecoder = ImageDecoder::Create( fileName );
            _taskCount += pDecoder->GetFrameCount();
            _taskNumberToFileIndex[int( _taskCount - 1 )] = int( i );
        }
        catch ( std::exception& e )
        {
            _error = e.what();
            _showError = true;
            return false;
        }        
    }*/
    return true;
}

int ImageReaderWindow::GetSerializedStringSize() const
{
    return PipelineElementWindow::GetSerializedStringSize()
        + FileListUser::GetSerializedStringSize();
}

std::string ImageReaderWindow::GetTaskName( size_t taskNumber ) const
{
    auto resOrErr = GetFrameSourceName(int(taskNumber));
    if ( resOrErr )
        return *resOrErr;
    else
        return resOrErr.error();
}

size_t ImageReaderWindow::GetTaskCount( bool )
{
    return _taskCount;
}

REGISTER_TOOLS_ITEM( ImageReaderWindow )

ACMB_GUI_NAMESPACE_END
