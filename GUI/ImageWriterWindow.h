#pragma once
#include "PipelineElementWindow.h"

ACMB_GUI_NAMESPACE_BEGIN

class ImageWriterWindow : public PipelineElementWindow
{
    std::string _workingDirectory;
    std::string _fileName;

    virtual IBitmapPtr ProcessBitmapFromPrimaryInput( IBitmapPtr pSource, size_t taskNumber ) override;

public:
    ImageWriterWindow( const Point& gridPos );

    virtual void DrawPipelineElementControls() override;

    std::vector<std::string> RunAllTasks();

    inline static const std::string icon = "\xef\x83\x87";
    inline static const std::string tooltip = "Save image";
    inline static constexpr uint8_t order = 2;
    virtual uint8_t GetMenuOrder() override { return order; }
};

ACMB_GUI_NAMESPACE_END

