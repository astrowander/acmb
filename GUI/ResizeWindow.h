#pragma once
#include "PipelineElementWindow.h"

ACMB_GUI_NAMESPACE_BEGIN

class ResizeWindow : public PipelineElementWindow
{
    Size _dstSize = { 1920, 1080 };

    virtual IBitmapPtr ProcessBitmapFromPrimaryInput( IBitmapPtr pSource, size_t taskNumber = 0 ) override;

public:

    ResizeWindow( const Point& gridPos );
    virtual void DrawPipelineElementControls() override;

    inline static const std::string icon = "\xef\x90\xa4";
    inline static const std::string tooltip = "Resize image";
    inline static constexpr uint8_t order = 6;

    virtual uint8_t GetMenuOrder() override { return order; }

    virtual void Serialize(std::ostream& out) override;
    virtual void Deserialize(std::istream& in) override;
};

ACMB_GUI_NAMESPACE_END
