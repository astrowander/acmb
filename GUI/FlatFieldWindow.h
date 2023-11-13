#pragma once
#include "PipelineElementWindow.h"

ACMB_GUI_NAMESPACE_BEGIN

class DivideImageWindow : public PipelineElementWindow
{
    float _intensity = 100.0f;

    virtual IBitmapPtr ProcessBitmapFromPrimaryInput( IBitmapPtr pSource, size_t taskNumber ) override;
public:

    DivideImageWindow( const Point& gridPos );
    virtual void DrawPipelineElementControls() override;
    virtual void Serialize(std::ostream& out) override;
    virtual void Deserialize(std::istream& in) override;

    SET_MENU_PARAMS( "\xef\x94\xa9", "Flat field", "Divide images on a flat field. By default the flat field is on the top of the tool, and the target images are on the left", 5 );
};

ACMB_GUI_NAMESPACE_END
