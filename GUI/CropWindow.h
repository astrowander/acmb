#pragma once
#include "PipelineElementWindow.h"
#include "SettingsInterpolationUser.h"

#include "./../Transforms/CropTransform.h"

ACMB_GUI_NAMESPACE_BEGIN

class CropWindow : public PipelineElementWindow, public SettingsInterpolationUser<CropTransform>
{
private:
    CropTransform::Settings _dstRect = { 0, 0, 10000, 10000 };

    virtual IBitmapPtr ProcessBitmapFromPrimaryInput( IBitmapPtr pSource, size_t taskNumber = 0 ) override;
    virtual Expected<void, std::string> GeneratePreviewBitmap() override;
    virtual Expected<Size, std::string> GetBitmapSize() override;
public:

    CropWindow(  );
    virtual void DrawPipelineElementControls() override;
    virtual void Serialize(std::ostream& out) const override;
    virtual bool Deserialize(std::istream& in) override;
    virtual int GetSerializedStringSize() const override;

    virtual void OnPreviewedFrameNumberChanged(int val) override;
    virtual void OnKeyframeCommited() override;

    SET_MENU_PARAMS( "\xef\x84\xa5", "Crop", "Crop image to an arbitrary rectangle", 7 );
};

ACMB_GUI_NAMESPACE_END
