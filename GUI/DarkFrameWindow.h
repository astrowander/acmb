#pragma once
#include "PipelineElementWindow.h"

ACMB_GUI_NAMESPACE_BEGIN

class DarkFrameWindow : public PipelineElementWindow
{
    float _multiplier = 1.0f;
    virtual IBitmapPtr ProcessBitmapFromPrimaryInput( IBitmapPtr pSource, size_t taskNumber ) override;

    Expected<float, std::string> AutoAdjustMultiplier();

    virtual Expected<void, std::string> GeneratePreviewTexture() override;

public:

    DarkFrameWindow( const Point& gridPos );
    virtual void DrawPipelineElementControls() override;
    virtual void Serialize( std::ostream& out ) const override;
    virtual void Deserialize( std::istream& in ) override;
    virtual int GetSerializedStringSize() const override;

    SET_MENU_PARAMS( "\xef\x81\xa8", "Dark frame", "Subtract a dark frame from the target images. By default the dark frame is on the top of the tool, and the target images are on the left", 4 );
};

ACMB_GUI_NAMESPACE_END
