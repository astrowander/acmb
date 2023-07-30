#pragma once
#include "PipelineElementWindow.h"

ACMB_GUI_NAMESPACE_BEGIN

class ImageWriterWindow : public PipelineElementWindow
{
    enum class FileFormat
    {
        Ppm,
        Tiff,
        Jpeg
    };

    std::string _workingDirectory;
    std::string _fileName;
    FileFormat _fileFormat = FileFormat::Ppm;

    virtual std::expected<IBitmapPtr, std::string> RunTask( size_t i ) override;
    
public:
    ImageWriterWindow( const ImVec2& pos, std::shared_ptr<Window> pParent );

    virtual void DrawPipelineElementControls() override;

    std::vector<std::string> RunAllTasks();
};

ACMB_GUI_NAMESPACE_END

