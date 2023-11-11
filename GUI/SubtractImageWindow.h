#pragma once
#include "PipelineElementWindow.h"

ACMB_GUI_NAMESPACE_BEGIN

class SubtractImageWindow : public PipelineElementWindow
{
    IBitmapPtr _pBitmapToSubtract;
    int _darkFrameIsOnTop = 1;

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

    virtual void Serialize(std::ostream& out) override;
    virtual void Deserialize(std::istream& in) override;

    virtual std::shared_ptr<PipelineElementWindow> GetPrimaryInput() override { return _darkFrameIsOnTop ? GetLeftInput() : GetTopInput(); }
};

ACMB_GUI_NAMESPACE_END
