#pragma once

#include "PipelineElementWindow.h"
#include "Texture.h"
ACMB_GUI_NAMESPACE_BEGIN

class ImageReaderWindow : public PipelineElementWindow
{
    std::string _workingDirectory;
    std::vector<std::string> _fileNames;
    int _selectedItemIdx = 0;
    std::unique_ptr<Texture> _pPreviewTexture;

    virtual Expected<IBitmapPtr, std::string> RunTask( size_t i ) override;
    virtual IBitmapPtr ProcessBitmapFromPrimaryInput( IBitmapPtr, size_t ) override { return nullptr; }

public:
    ImageReaderWindow( const Point& gridPos );
    virtual void DrawPipelineElementControls() override;
    virtual void Serialize(std::ostream& out) const override;
    virtual void Deserialize(std::istream& in) override;
    virtual int GetSerializedStringSize() const override;
    virtual size_t GetTaskCount() override;

    std::string GetTaskName( size_t taskNumber ) const override;

    SET_MENU_PARAMS( "\xef\x87\x85", "Import", "Choose images to import and pass them to another tools", 1 );
};

ACMB_GUI_NAMESPACE_END
