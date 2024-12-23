#pragma once
#include "PipelineElementWindow.h"
#include "../Transforms/BitmapHealer.h"

ACMB_GUI_NAMESPACE_BEGIN

class BitmapHealerWindow : public PipelineElementWindow
{
    virtual IBitmapPtr ProcessBitmapFromPrimaryInput( IBitmapPtr pSource, size_t taskNumber = 0 ) override;
    virtual Expected<void, std::string> GeneratePreviewBitmap() override;
public:
    BitmapHealerWindow( const Point& gridPos );
    virtual void DrawPipelineElementControls() override;
    virtual void Serialize( std::ostream& out ) const override;
    virtual bool Deserialize( std::istream& in ) override;
    virtual int GetSerializedStringSize() const override;

    SET_MENU_PARAMS( "\xef\x83\x90", "Healer", "Heal spots in the image", 15 );

private:
    BitmapHealer::Settings _patches;
    int _currentPatch = 0;
    //float _x = 0.0f;
    //float _y = 0.0f;
};

ACMB_GUI_NAMESPACE_END
