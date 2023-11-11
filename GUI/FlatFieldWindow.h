#pragma once
#include "PipelineElementWindow.h"

ACMB_GUI_NAMESPACE_BEGIN

class DivideImageWindow : public PipelineElementWindow
{
    IBitmapPtr _pFlatField;
    float _intensity = 100.0f;
    int _flatFieldIsOnTop = 1;

    //virtual std::expected<IBitmapPtr, std::string> RunTask( size_t i ) override;
    virtual IBitmapPtr ProcessBitmapFromPrimaryInput( IBitmapPtr pSource, size_t taskNumber ) override;
public:

    DivideImageWindow( const Point& gridPos );
    virtual void DrawPipelineElementControls() override;

    inline static const std::string icon = "\xef\x94\xa9";
    inline static const std::string tooltip = "Apply flat field";
    inline static constexpr uint8_t order = 5;

    virtual uint8_t GetMenuOrder() override { return order; }
    virtual void Serialize(std::ostream& out) override;
    virtual void Deserialize(std::istream& in) override;

    virtual std::shared_ptr<PipelineElementWindow> GetPrimaryInput() override { return _flatFieldIsOnTop ? GetLeftInput() : GetTopInput(); }
};

ACMB_GUI_NAMESPACE_END
