#pragma once
#include "PipelineElementWindow.h"

ACMB_GUI_NAMESPACE_BEGIN

class ResizeWindow : public PipelineElementWindow
{
    Size _dstSize = { 1920, 1080 };

    virtual IBitmapPtr ProcessBitmapFromPrimaryInput( IBitmapPtr pSource, size_t taskNumber = 0 ) override;
    virtual Expected<void, std::string> GeneratePreviewBitmap() override;
    virtual Expected<Size, std::string> GetBitmapSize() override;
public:

    ResizeWindow( const Point& gridPos );
    virtual void DrawPipelineElementControls() override;
    virtual void Serialize(std::ostream& out) const override;
    virtual void Deserialize(std::istream& in) override;
    virtual int GetSerializedStringSize() const override;

    SET_MENU_PARAMS( "\xef\x90\xa4", "Resize", "Choose image to arbitrary size", 6 );
};

ACMB_GUI_NAMESPACE_END
