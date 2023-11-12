#pragma once
#include "PipelineElementWindow.h"

ACMB_GUI_NAMESPACE_BEGIN

class SubtractImageWindow : public PipelineElementWindow
{
    virtual IBitmapPtr ProcessBitmapFromPrimaryInput( IBitmapPtr pSource, size_t taskNumber ) override;

public:

    SubtractImageWindow( const Point& gridPos );
    virtual void DrawPipelineElementControls() override;

    inline static const std::string icon = "\xef\x81\xa8";
    inline static const std::string tooltip = "Subtract dark frame";
    inline static constexpr uint8_t order = 4;

    virtual uint8_t GetMenuOrder() override
    {
        return order;
    }
};

ACMB_GUI_NAMESPACE_END
