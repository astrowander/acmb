#include "ImageReaderWindow.h"
#include "MainWindow.h"
#include "Serializer.h"
#include "ImGuiHelpers.h"
#include "./../Codecs/imagedecoder.h"
#include "imgui/imgui_internal.h"

#include <sstream>
#include "DarkFrameWindow.h"

ACMB_GUI_NAMESPACE_BEGIN



ImageReaderWindow::ImageReaderWindow(  )
    : PipelineElementWindow( "Import Images" )
    , FileListUser(this)
{
}

void ImageReaderWindow::DrawPipelineElementControls()
{
    FileListUser::DrawControls();
}

void ImageReaderWindow::ResetTasks(){
    PipelineElementWindow::ResetTasks();
}

Expected<IBitmapPtr, std::string> ImageReaderWindow::GeneratePreviewBitmap(bool forNextElement, bool fullSize)
{
    Size size;
    if ( fullSize )
    {
        auto sizeOrErr = GetFrameSize( _currentFrameIdx );
        if ( !sizeOrErr )
            return unexpected( sizeOrErr.error() );
        size = *sizeOrErr;
    }
    else
    {
        const RectF rect = MainWindow::GetInstance().GetImageRegionAvail();
        size = Size{ int( rect.width ), int( rect.height ) };
    }

    auto resOrErr = ReadFramePreview( _currentFrameIdx, size );
    if ( resOrErr )
    {
        return *resOrErr;
    }

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
    return GetFrameSize( _currentFrameIdx );
}

void ImageReaderWindow::OnSelectedFrameChanged(int idx)
{
    OnPreviewedFrameNumberChanged(idx);
}

void ImageReaderWindow::OnFileListChanged()
{    
    _showPreview = false;
    _taskCount = GetTotalFrameCount();
    ImGui::CloseCurrentPopup();
    ResetProgress(PropagationDir::Forward);

    FileListUser::OnFileListChanged();

    PipelineElementWindow* pElement = this;
    while ( pElement && pElement->GetOutput() )
    {
        pElement = pElement->GetOutput().get();
    }

    pElement->GetTaskCount(true);
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
    if ( _taskCount == 0 )
    {
        _taskCount = GetTotalFrameCount();
    }

    return _taskCount;
}

REGISTER_TOOLS_ITEM( ImageReaderWindow )

ACMB_GUI_NAMESPACE_END
