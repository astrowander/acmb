#pragma once

#include "PipelineElementWindow.h"
#include "SettingsInterpolationUser.h"

#include "./../Transforms/LevelsTransform.h"

ACMB_GUI_NAMESPACE_BEGIN

class LevelsWindow : public PipelineElementWindow, public SettingsInterpolationUser<LevelsTransform>
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

    virtual void OnPreviewedFrameNumberChanged(int number) override;
    virtual void OnKeyframeCommited() override;

    SET_MENU_PARAMS( "\xef\x82\x80", "Levels", "Adjust levels of the image", 9 );

private:
    LevelsTransform::Settings _levelsSettings;
};

ACMB_GUI_NAMESPACE_END