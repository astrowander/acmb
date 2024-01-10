#pragma once

#include "PipelineElementWindow.h"

ACMB_GUI_NAMESPACE_BEGIN

class LevelsWindow : public PipelineElementWindow
{
    IBitmapPtr _pCachedPreview;
    std::vector<uint32_t> _channelHistograms[4];

    virtual IBitmapPtr ProcessBitmapFromPrimaryInput( IBitmapPtr pSource, size_t taskNumber = 0 ) override;

public:
    LevelsWindow( const Point& gridPos );
    virtual void DrawPipelineElementControls() override;
    virtual void Serialize( std::ostream& out ) const override;
    virtual void Deserialize( std::istream& in ) override;
    virtual int GetSerializedStringSize() const override;

    SET_MENU_PARAMS( "\xef\x82\x80", "Levels", "Adjust levels of the image", 9 );
};

ACMB_GUI_NAMESPACE_END