#pragma once
#include "FileListUser.h"

ACMB_GUI_NAMESPACE_BEGIN

class DarkFrameWindow : public PipelineElementWindow, public FileListUser
{
    float _multiplier = 1.0f;
    virtual IBitmapPtr ProcessBitmapFromPrimaryInput( IBitmapPtr pSource, size_t taskNumber ) override;

    virtual Expected<void, std::string> GeneratePreviewBitmap() override;

    virtual void OnSelectedFrameChanged(int idx) override {}

    virtual std::string GetWindowName() const override;
    virtual std::string GetFileFilters() const;

public:

    DarkFrameWindow( const Point& gridPos );
    virtual void DrawPipelineElementControls() override;
    virtual void Serialize( std::ostream& out ) const override;
    virtual bool Deserialize( std::istream& in ) override;
    virtual int GetSerializedStringSize() const override;

    SET_MENU_PARAMS( "\xef\x81\xa8", "Dark frame", "Subtract a dark frame from the target images. By default the dark frame is on the top of the tool, and the target images are on the left", 4 );
};

ACMB_GUI_NAMESPACE_END
