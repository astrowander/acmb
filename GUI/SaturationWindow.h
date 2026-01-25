#pragma once
#include "PipelineElementWindow.h"
#include "SettingsInterpolationUser.h"

#include "./../Transforms/SaturationTransform.h"

ACMB_GUI_NAMESPACE_BEGIN

class SaturationWindow : public PipelineElementWindow, public SettingsInterpolationUser<SaturationTransform>
{
    virtual IBitmapPtr ProcessBitmapFromPrimaryInput( IBitmapPtr pSource, size_t taskNumber = 0 ) override;
    virtual Expected<void, std::string> GeneratePreviewBitmap() override;
public:
    SaturationWindow(  );
    virtual void DrawPipelineElementControls() override;
    virtual void Serialize( std::ostream& out ) const override;
    virtual bool Deserialize( std::istream& in ) override;
    virtual int GetSerializedStringSize() const override;

    SET_MENU_PARAMS( "\xef\x81\x82", "Saturation", "Adjust saturation of the image", 10 );

    virtual void OnPreviewedFrameNumberChanged(int number) override;
    virtual void OnKeyframeCommited() override;

private:
    SaturationTransform::Settings _saturationSettings = 1.0f;
};

ACMB_GUI_NAMESPACE_END