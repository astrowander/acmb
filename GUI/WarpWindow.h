#pragma once
#include "PipelineElementWindow.h"
#include "../Transforms/WarpTransform.h"

ACMB_GUI_NAMESPACE_BEGIN

class WarpWindow : public PipelineElementWindow
{
    virtual IBitmapPtr ProcessBitmapFromPrimaryInput( IBitmapPtr pSource, size_t taskNumber = 0 ) override;
    virtual Expected<IBitmapPtr, std::string> GeneratePreviewBitmap(bool forNextElement, bool fullSize) override;
public:
    WarpWindow(  );
    virtual void DrawPipelineElementControls() override;
    virtual void Serialize( std::ostream& out ) const override;
    virtual bool Deserialize( std::istream& in ) override;
    virtual int GetSerializedStringSize() const override;

    SET_MENU_PARAMS( "\xef\x94\xbb", "Warp", "Deform an image with Bezier curves", 14 );

private:
    WarpTransform::Settings _settings;
    float _x = 0.0f;
    float _y = 0.0f;
};

ACMB_GUI_NAMESPACE_END
