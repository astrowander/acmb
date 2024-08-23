#pragma once

#include "PipelineElementWindow.h"

ACMB_GUI_NAMESPACE_BEGIN
 
class ConverterWindow : public PipelineElementWindow
{
    PixelFormat _dstPixelFormat = PixelFormat::RGB24;

    virtual IBitmapPtr ProcessBitmapFromPrimaryInput( IBitmapPtr pSource, size_t taskNumber = 0 ) override;
    virtual Expected<void, std::string> GeneratePreviewBitmap() override;

public:

    ConverterWindow( const Point& gridPos );
    virtual void DrawPipelineElementControls() override;
    virtual void Serialize(std::ostream& out) const override;
    virtual bool Deserialize(std::istream& in) override;
    virtual int GetSerializedStringSize() const override;

    SET_MENU_PARAMS( "\xef\x86\xb8", "Converter", "Convert image to another pixel format", 8 );
};

ACMB_GUI_NAMESPACE_END
