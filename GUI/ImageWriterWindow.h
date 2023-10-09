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
    ImageWriterWindow( const Point& gridPos );

    virtual void DrawPipelineElementControls() override;

    std::vector<std::string> RunAllTasks();

    inline static const std::string icon = "\xef\x83\x87";
    inline static const std::string tooltip = "Save image";
    inline static constexpr uint8_t order = 2;

    virtual uint8_t GetMenuOrder() override
    {
        return order;
    }
};

ACMB_GUI_NAMESPACE_END

