#pragma once
#include "ImGuiHelpers.h"
#include "PipelineElementWindow.h"

ACMB_NAMESPACE_BEGIN
class ImageDecoder;
ACMB_NAMESPACE_END

ACMB_GUI_NAMESPACE_BEGIN

class FileListUser
{    
    PipelineElementWindow* _pHost = nullptr;

    mutable std::shared_ptr<ImageDecoder> _pDecoder;

    mutable int _currentFileNumber = -1; // -1 means no current    
    int _totalFrameCount = 0;
    std::string _workingDirectory = ".";
    std::vector<int> _frameIndicesStartsWith;
    std::vector<std::string> _fileNames;
    IBitmapPtr _pStackedFrames;
    ImageParams _imageParams;

private:
    void PrepareFrameForReading(int idx) const;
    Expected<void, std::string> AddFile(const std::string& fileName);
    Expected<void, std::string> AddFiles(const std::vector<std::string>& fileNames);

    virtual void OnSelectedFrameChanged(int idx) = 0;
    

    virtual std::string GetWindowName() const = 0;
    virtual std::string GetFileFilters() const = 0;
protected:
    FileListUser( PipelineElementWindow* pHost ) : _pHost(pHost) {}
    ~FileListUser() = default;

    Expected<Size, std::string> GetFrameSize(int idx) const;
    Expected<IBitmapPtr, std::string> ReadFrame(int idx) const;
    Expected<IBitmapPtr, std::string> ReadFramePreview(int idx, Size size) const;
    Expected<std::string, std::string> GetFrameSourceName(int idx) const;

    int GetTotalFrameCount() const;

    void DrawControls();
    void Serialize(std::ostream& out) const;
    bool Deserialize(std::istream& in, int& remainingBytes);
    int GetSerializedStringSize() const;

    IBitmapPtr GetBitmapOfStackedFrames();
    virtual void OnFileListChanged();

    void CleanUp();
};

ACMB_GUI_NAMESPACE_END
