#pragma once

#include "FileListUser.h"

ACMB_NAMESPACE_BEGIN
class ImageDecoder;
ACMB_NAMESPACE_END

ACMB_GUI_NAMESPACE_BEGIN

class ImageReaderWindow : public PipelineElementWindow, public FileListUser
{
    virtual Expected<IBitmapPtr, std::string> RunTask( size_t i ) override;
    virtual IBitmapPtr ProcessBitmapFromPrimaryInput( IBitmapPtr, size_t ) override { return nullptr; }

    virtual Expected<IBitmapPtr, std::string> GeneratePreviewBitmap(bool forNextElement, bool fullSize) override;
    virtual Expected<Size, std::string> GetBitmapSize() override;

    virtual void OnSelectedFrameChanged(int idx) override;

    virtual std::string GetWindowName() const override;
    virtual std::string GetFileFilters() const;

    virtual int GetPreviewedFrameNumber() const override
    { 
        return _currentFrameIdx;
    }

    virtual void SetPreviewedFrameNumber( int val ) override
    {
        if ( val >= 0 && val < GetTotalFrameCount() )
        {
            _currentFrameIdx = val;
        }
    }

public:
    ImageReaderWindow(  );
    virtual void DrawPipelineElementControls() override;
    virtual void Serialize(std::ostream& out) const override;
    virtual bool Deserialize(std::istream& in) override;
    virtual int GetSerializedStringSize() const override;
    virtual size_t GetTaskCount(bool update = false) override;

    virtual void ResetTasks() override;

    virtual void OnFileListChanged() override;

    std::string GetTaskName( size_t taskNumber ) const override;

    SET_MENU_PARAMS( "\xef\x87\x85", "Import", "Choose images to import and pass them to another tools", 1 );
};

ACMB_GUI_NAMESPACE_END
