#pragma once

#include "PipelineElementWindow.h"

ACMB_GUI_NAMESPACE_BEGIN

class LevelsWindow : public PipelineElementWindow
{
    
    //std::vector<uint32_t> _channelHistograms[4];
    virtual IBitmapPtr ProcessBitmapFromPrimaryInput( IBitmapPtr pSource, size_t taskNumber = 0 ) override;
    virtual Expected<void, std::string> GeneratePreviewBitmap() override;

    virtual Expected<void, std::string> AutoAdjustLevels();
public:
    
    struct LevelsSettings
    {
        float min = 0.0f;
        float gamma = 1.0f;
        float max = 1.0f;
    };

    LevelsWindow( const Point& gridPos );
    virtual void DrawPipelineElementControls() override;
    virtual void Serialize( std::ostream& out ) const override;
    virtual bool Deserialize( std::istream& in ) override;
    virtual int GetSerializedStringSize() const override;

    SET_MENU_PARAMS( "\xef\x82\x80", "Levels", "Adjust levels of the image", 9 );

private:
    std::array<LevelsSettings, 4> _levelsSettings;
    bool _adjustChannels = false;
};

ACMB_GUI_NAMESPACE_END