#pragma once
#include "PipelineElementWindow.h"

ACMB_GUI_NAMESPACE_BEGIN

class DeflickerWindow : public PipelineElementWindow
{
    int _framesPerChunk = 10;
    int _iterations = 3;
    std::vector<IBitmapPtr> _bitmaps;

    virtual Expected<IBitmapPtr, std::string> RunTask( size_t i ) override;
    virtual IBitmapPtr ProcessBitmapFromPrimaryInput( IBitmapPtr, size_t ) override {
        return nullptr;
    }
    virtual Expected<void, std::string> GeneratePreviewBitmap() override;

public:

    DeflickerWindow( const Point& gridPos );
    virtual void DrawPipelineElementControls() override;
    virtual void Serialize( std::ostream& out ) const override;
    virtual bool Deserialize( std::istream& in ) override;
    virtual int GetSerializedStringSize() const override;

    SET_MENU_PARAMS( "\xef\x89\x8e", "Deflicker", "Remove flickering from the images", 16);
};

ACMB_GUI_NAMESPACE_END