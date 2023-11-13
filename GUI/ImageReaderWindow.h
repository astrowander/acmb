#pragma once

#include "PipelineElementWindow.h"

ACMB_GUI_NAMESPACE_BEGIN

class ImageReaderWindow : public PipelineElementWindow
{
    std::string _workingDirectory;
    std::vector<std::string> _fileNames;
    int _selectedItemIdx = 0;

    virtual std::expected<IBitmapPtr, std::string> RunTask( size_t i ) override;
    virtual IBitmapPtr ProcessBitmapFromPrimaryInput( IBitmapPtr pSource, size_t taskNumber ) override { return nullptr; }

public:
    ImageReaderWindow( const Point& gridPos );
    virtual void DrawPipelineElementControls() override;
    virtual void Serialize(std::ostream& out) override;
    virtual void Deserialize(std::istream& in) override;

    SET_MENU_PARAMS( "\xef\x87\x85", "Import", "Choose images to import and pass them to another tools", 1 );
};

ACMB_GUI_NAMESPACE_END
