#pragma once

#include "PipelineElementWindow.h"

ACMB_GUI_NAMESPACE_BEGIN
 
class ConverterWindow : public PipelineElementWindow
{
    PixelFormat _dstPixelFormat = PixelFormat::RGB24;

    virtual IBitmapPtr ProcessBitmapFromPrimaryInput( IBitmapPtr pSource, size_t taskNumber = 0 ) override;

public:

    ConverterWindow( const Point& gridPos );
    virtual void DrawPipelineElementControls() override;

    inline static const std::string icon = "\xef\x86\xb8";
    inline static const std::string tooltip = "Converter";
    inline static constexpr uint8_t order = 8;

    virtual uint8_t GetMenuOrder() override { return order; }

    virtual void Serialize(std::ostream& out) override;
    virtual void Deserialize(std::istream& in) override;
};

ACMB_GUI_NAMESPACE_END
