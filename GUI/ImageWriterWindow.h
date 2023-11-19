#pragma once
#include "PipelineElementWindow.h"

ACMB_GUI_NAMESPACE_BEGIN

class ImageWriterWindow : public PipelineElementWindow
{
    std::string _workingDirectory;
    std::string _fileName;
    std::string _formatList;    
    int _formatId = 0;
    bool _keepOriginalFileName = false;
    
    virtual IBitmapPtr ProcessBitmapFromPrimaryInput( IBitmapPtr pSource, size_t taskNumber ) override;

public:
    ImageWriterWindow( const Point& gridPos );
    virtual void DrawPipelineElementControls() override;
    virtual void Serialize( std::ostream& out ) const override;
    virtual void Deserialize( std::istream& in ) override;
    virtual int GetSerializedStringSize() const override;

    std::vector<std::string> RunAllTasks();

    SET_MENU_PARAMS( "\xef\x95\xad", "Export", "Choose a file or a directory where to save the results", 2 );
};

ACMB_GUI_NAMESPACE_END

