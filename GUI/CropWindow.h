#pragma once
#include "PipelineElementWindow.h"

ACMB_GUI_NAMESPACE_BEGIN

class CropWindow : public PipelineElementWindow
{
    Rect _dstRect = { 0, 0, 1000, 1000 };

    virtual std::expected<IBitmapPtr, std::string> RunTask( size_t i ) override;

public:

    CropWindow( const Point& gridPos );
    virtual void DrawPipelineElementControls() override;

    inline static const std::string icon = "\xef\x84\xa5";
    inline static const std::string tooltip = "Crop image";
    inline static constexpr uint8_t order = 7;

    virtual uint8_t GetMenuOrder() override
    {
        return order;
    }
};

ACMB_GUI_NAMESPACE_END
