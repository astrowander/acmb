#pragma once
#include "PipelineElementWindow.h"

ACMB_GUI_NAMESPACE_BEGIN

class DarkFrameWindow : public PipelineElementWindow
{
    virtual IBitmapPtr ProcessBitmapFromPrimaryInput( IBitmapPtr pSource, size_t taskNumber ) override;

public:

    DarkFrameWindow( const Point& gridPos );
    virtual void DrawPipelineElementControls() override;

    SET_MENU_PARAMS( "\xef\x81\xa8", "Dark frame", "Subtract a dark frame from the target images. By default the dark frame is on the top of the tool, and the target images are on the left", 4 );
};

ACMB_GUI_NAMESPACE_END
