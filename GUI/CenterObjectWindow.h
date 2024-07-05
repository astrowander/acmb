#pragma once
#include "PipelineElementWindow.h"

ACMB_GUI_NAMESPACE_BEGIN

class CenterObjectWindow : public PipelineElementWindow
{
    Size _dstSize;
    float _threshold = 25.0f;

    virtual IBitmapPtr ProcessBitmapFromPrimaryInput( IBitmapPtr pSource, size_t taskNumber = 0 ) override;
    virtual Expected<void, std::string> GeneratePreviewBitmap() override;

public:

    CenterObjectWindow( const Point& gridPos );
    virtual void DrawPipelineElementControls() override;
    virtual void Serialize( std::ostream& out ) const override;
    virtual void Deserialize( std::istream& in ) override;
    virtual int GetSerializedStringSize() const override;

    SET_MENU_PARAMS( "\xef\x81\x9b", "Centering", "Move image of the object to the center and crop it", 11 );
};

ACMB_GUI_NAMESPACE_END
