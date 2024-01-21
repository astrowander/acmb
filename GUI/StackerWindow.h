#pragma once
#include "PipelineElementWindow.h"

ACMB_GUI_NAMESPACE_BEGIN

class StackerWindow : public PipelineElementWindow
{
    StackMode _stackMode = StackMode::Light;
    float _threshold = 25.0f;
    bool _autoContrast = false;

    virtual Expected<IBitmapPtr, std::string> RunTask( size_t i ) override;
    virtual IBitmapPtr ProcessBitmapFromPrimaryInput( IBitmapPtr, size_t ) override { return nullptr; }
    virtual Expected<void, std::string> GeneratePreviewBitmap() override;

public:

    StackerWindow( const Point& gridPos );
    virtual void DrawPipelineElementControls() override;
    virtual void Serialize(std::ostream& out) const override;
    virtual void Deserialize(std::istream& in) override;
    virtual int GetSerializedStringSize() const override;

    virtual void ResetTasks() override;

    SET_MENU_PARAMS( "\xef\x97\xbd", "Stack", "Sum up a group of frames to one image", 3 );
};

ACMB_GUI_NAMESPACE_END
