#include "FileListUser.h"
#include "FileDialog.h"
#include "Serializer.h"

#include "./../Codecs/imagedecoder.h"
#include "./../Registrator/stacker.h"

#include <algorithm>

ACMB_GUI_NAMESPACE_BEGIN

void FileListUser::PrepareFrameForReading(int frameIdx) const
{
    auto it = std::upper_bound(_frameIndicesStartsWith.begin(), _frameIndicesStartsWith.end(), frameIdx);
    if ( it == _frameIndicesStartsWith.begin() )
    {
        _lastDecodedFileIdx = -1;
        _pDecoder.reset();
        return;
    }

    int fileIdx = int(std::distance(_frameIndicesStartsWith.begin(), it)) - 1;

    if ( _pDecoder && _lastDecodedFileIdx == fileIdx )
        return;

    _lastDecodedFileIdx = fileIdx;
    _pDecoder = ImageDecoder::Create( _fileNames[fileIdx] );
}

Expected<void, std::string> FileListUser::AddFile(const std::string& fileName)
{
    try
    {
        _fileNames.push_back(fileName);

        int indicesStartsWith = _frameIndicesStartsWith.empty() ? 0 : _frameIndicesStartsWith.back();
        if ( _pDecoder )
            indicesStartsWith += _pDecoder->GetFrameCount();

        _frameIndicesStartsWith.push_back(indicesStartsWith);
        _pDecoder = ImageDecoder::Create(fileName);
        _totalFrameCount += _pDecoder->GetFrameCount();
        if ( _imageParams.GetPixelFormat() == PixelFormat::Unspecified )
            _imageParams = *_pDecoder;

        return {};
    }
    catch ( std::exception& e )
    {
        return unexpected(e.what());
    }
}

/*Expected<void, std::string> FileListUser::AddFiles(const std::vector<std::string>& fileNames)
{
    PrepareFrameForReading( GetTotalFrameCount() - 1 );

    for ( const auto& fileName : fileNames )
    {
        
    }
    return {};
}*/

Expected<Size, std::string> FileListUser::GetFrameSize(int idx) const
{
    std::lock_guard<std::mutex> lock(_mutex);

    PrepareFrameForReading(idx);
    if ( !_pDecoder )
        return unexpected( "FileListUser::GetFrameSize: No decoder" );

    return Size{ int(_pDecoder->GetWidth()), int(_pDecoder->GetHeight()) };
}

Expected<IBitmapPtr, std::string>  FileListUser::ReadFrame(int idx) const
{
    std::lock_guard<std::mutex> lock(_mutex);

    PrepareFrameForReading(idx);
    if ( !_pDecoder )
        return unexpected( "FileListUser::ReadFrame: No decoder" );

    return _pDecoder->ReadBitmap( idx - _frameIndicesStartsWith[_lastDecodedFileIdx] );
}

Expected<IBitmapPtr, std::string> FileListUser::ReadFramePreview(int idx, Size size) const
{
    std::lock_guard<std::mutex> lock(_mutex);

    PrepareFrameForReading(idx);
    if ( !_pDecoder )
        return unexpected("FileListUser::ReadFrame: No decoder");

    return _pDecoder->ReadPreview(size);
}

Expected<std::string, std::string> FileListUser::GetFrameSourceName(int idx) const
{
    std::lock_guard<std::mutex> lock(_mutex);

    PrepareFrameForReading(idx);
    if ( !_pDecoder )
        return unexpected( "FileListUser::GetFrameSourceName: No decoder" );

    return _fileNames[_currentFrameIdx];
}

int FileListUser::GetTotalFrameCount() const
{
    return _totalFrameCount;
}

void FileListUser::DrawControls()
{
    const auto& style = ImGui::GetStyle();
    const float itemWidth = PipelineElementWindow::cElementWidth - 2.0f * style.WindowPadding.x;

    if ( ImGui::BeginListBox("##ImageList", { itemWidth, 110 }) )
    {
        for ( int i = 0; i < int(_fileNames.size()); ++i )
        {
            const bool is_selected = (_currentFrameIdx == i);
            const std::string shortName = _fileNames[i].substr(_fileNames[i].find_last_of("\\/") + 1);
            if ( ImGui::Selectable(shortName.c_str(), is_selected) )
            {
                OnSelectedFrameChanged(_frameIndicesStartsWith[i]);
            }
            // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
            if ( is_selected )
                ImGui::SetItemDefaultFocus();
        }
        ImGui::EndListBox();
    }

    ImGui::Text("%d frames in %d files", _totalFrameCount, int(_fileNames.size()));

    auto fileDialog = FileDialog::Instance();
    const auto openDialogName = "SelectImagesDialog##" + GetWindowName();

    UI::Button("Select Images", { itemWidth, 0 }, [&]
    {

        static auto filters = GetFileFilters();
        fileDialog.OpenDialog(openDialogName, "Select Images", filters.c_str(), _workingDirectory.c_str(), 0);
    }, "Add images to the importing list", _pHost);

    UI::Button("Clear List", { itemWidth, 0 }, [&]
    {
        _fileNames.clear();
        {            
            CleanUp();
            OnFileListChanged();
        }
    }, "Delete all images from the importing list", _pHost);

    if ( fileDialog.Display(openDialogName, {}, { 300 * PipelineElementWindow::cMenuScaling, 200 * PipelineElementWindow::cMenuScaling }) )
    {
        _workingDirectory = fileDialog.GetCurrentPath() + "/";
        // action if OK
        if ( fileDialog.IsOk() )
        {
            const auto selection = fileDialog.GetSelection();
            for ( const auto& it : selection )
            {
                const auto path = _workingDirectory + it.first;
                AddFile(path);
            }
        }
        OnFileListChanged();
        // close
        fileDialog.Close();
    }
}

void FileListUser::Serialize(std::ostream& out) const
{
    gui::Serialize(_fileNames, out);
    gui::Serialize(_frameIndicesStartsWith, out);
    gui::Serialize(_totalFrameCount, out);
    gui::Serialize(_currentFrameIdx, out);
    gui::Serialize(_workingDirectory, out);
}

bool FileListUser::Deserialize(std::istream& in, int& remainingBytes)
{
    _fileNames = gui::Deserialize<std::vector<std::string>>( in, remainingBytes );
    _frameIndicesStartsWith = gui::Deserialize<std::vector<int>>( in, remainingBytes );
    _totalFrameCount = gui::Deserialize<int>( in, remainingBytes );
    _currentFrameIdx = gui::Deserialize<int>( in, remainingBytes );
    _workingDirectory = gui::Deserialize<std::string>( in, remainingBytes );
    return true;
}

int FileListUser::GetSerializedStringSize() const
{
    return gui::GetSerializedStringSize( _fileNames )
        + gui::GetSerializedStringSize( _frameIndicesStartsWith )
        + gui::GetSerializedStringSize( _totalFrameCount )
        + gui::GetSerializedStringSize( _currentFrameIdx )
        + gui::GetSerializedStringSize( _workingDirectory );
}

IBitmapPtr FileListUser::GetBitmapOfStackedFrames()
{
    if ( _pStackedFrames )
        return _pStackedFrames;
    
    std::lock_guard<std::mutex> lock(_mutex);

    PrepareFrameForReading(0);
    Stacker stacker(_imageParams, acmb::StackMode::DarkOrFlat);
    for ( int i = 0; i < _totalFrameCount; ++i )
        stacker.AddBitmap( ReadFrame(i).value_or( nullptr ) );

    _pStackedFrames = stacker.GetResult();
    return _pStackedFrames;
}

void FileListUser::OnFileListChanged()
{
    OnSelectedFrameChanged( GetTotalFrameCount() - 1 );
}

void FileListUser::CleanUp()
{
    std::lock_guard<std::mutex> lock(_mutex);

    _pStackedFrames = nullptr;
    _totalFrameCount = 0;
    _frameIndicesStartsWith.clear();
    _currentFrameIdx = -1;
    _pDecoder = nullptr;
    _imageParams = {};
}


ACMB_GUI_NAMESPACE_END