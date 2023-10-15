#pragma once
#include "PipelineElementWindow.h"

ACMB_GUI_NAMESPACE_BEGIN

class StackerWindow : public PipelineElementWindow
{
    StackMode _stackMode = StackMode::Light;
    //bool _enableCuda = false;

    virtual std::expected<IBitmapPtr, std::string> RunTask( size_t i ) override;

public:

    StackerWindow( const Point& gridPos );
    virtual void DrawPipelineElementControls() override;

    inline static const std::string icon = "\xef\x97\xbd";
    inline static const std::string tooltip = "Stack images";
    inline static constexpr uint8_t order = 3;

    virtual uint8_t GetMenuOrder() override { return order; }

    virtual void Serialize(std::ostream& out) override;
    virtual void Deserialize(std::istream& in) override;
};

ACMB_GUI_NAMESPACE_END
