#pragma once
#include "PipelineElementWindow.h"

ACMB_GUI_NAMESPACE_BEGIN

class FilteringWindow : public PipelineElementWindow
{
    enum class FilterMode
    {
        FrameCount,
        Percentage,
        QualityThreshold
    };

    FilterMode _filterMode = FilterMode::FrameCount;
    int _frameCount = 0;
    float _percentage = 0.0f;
    float _qualityThreshold = 0.0f;

    std::vector<IBitmapPtr> _sortedFrames;

    virtual Expected<IBitmapPtr, std::string> RunTask( size_t i ) override;
    virtual IBitmapPtr ProcessBitmapFromPrimaryInput( IBitmapPtr, size_t ) override { return nullptr; }
    virtual Expected<void, std::string> GeneratePreviewBitmap() override;

    Expected<void, std::string> SortFrames();

public:
    FilteringWindow( const Point& gridPos );

    virtual void DrawPipelineElementControls() override;
    virtual void Serialize( std::ostream& out ) const override;
    virtual bool Deserialize( std::istream& in ) override;
    virtual int GetSerializedStringSize() const override;

    virtual void ResetTasks() override;

    SET_MENU_PARAMS( "\xef\x85\xa3", "Sort Frames", "Sort frames by quality and output the bests", 13 );
};

ACMB_GUI_NAMESPACE_END
