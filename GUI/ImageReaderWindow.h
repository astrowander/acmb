#pragma once

#include "PipelineElementWindow.h"

ACMB_GUI_NAMESPACE_BEGIN

class ImageReaderWindow : public PipelineElementWindow
{
    std::string _workingDirectory;
    std::vector<std::string> _fileNames;
    int _selectedItemIdx = 0;

    virtual std::expected<IBitmapPtr, std::string> RunTask( size_t i ) override;

public:
    ImageReaderWindow( const ImVec2& pos, std::shared_ptr<Window> pParent );    
    virtual void DrawPipelineElementControls() override;
};

ACMB_GUI_NAMESPACE_END
