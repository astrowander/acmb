#pragma once

#include "PipelineElementWindow.h"
#include "./../Transforms/LevelsTransform.h"
ACMB_GUI_NAMESPACE_BEGIN

class LevelsWindow : public PipelineElementWindow
{
    virtual IBitmapPtr ProcessBitmapFromPrimaryInput( IBitmapPtr pSource, size_t taskNumber = 0 ) override;
    virtual Expected<void, std::string> GeneratePreviewBitmap() override;

    virtual Expected<void, std::string> AutoAdjustLevels();
public:

    LevelsWindow( const Point& gridPos );
    virtual void DrawPipelineElementControls() override;
    virtual void Serialize( std::ostream& out ) const override;
    virtual bool Deserialize( std::istream& in ) override;
    virtual int GetSerializedStringSize() const override;

    SET_MENU_PARAMS( "\xef\x82\x80", "Levels", "Adjust levels of the image", 9 );

private:
    LevelsTransform::Settings _levelsSettings;
};

ACMB_GUI_NAMESPACE_END