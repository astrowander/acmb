#pragma once
#include "PipelineElementWindow.h"

ACMB_GUI_NAMESPACE_BEGIN

class CropWindow : public PipelineElementWindow
{
    Rect _dstRect = { 0, 0, 1000, 1000 };

    virtual IBitmapPtr ProcessBitmapFromPrimaryInput( IBitmapPtr pSource, size_t taskNumber = 0 ) override;

public:

    CropWindow( const Point& gridPos );
    virtual void DrawPipelineElementControls() override;
    virtual void Serialize(std::ostream& out) override;
    virtual void Deserialize(std::istream& in) override;

    SET_MENU_PARAMS( "\xef\x84\xa5", "Crop", "Crop image to an arbitrary rectangle", 7 );
};

ACMB_GUI_NAMESPACE_END
