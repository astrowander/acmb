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
    ImageReaderWindow( const Point& gridPos );
    virtual void DrawPipelineElementControls() override;

    inline static const std::string icon = "\xef\x87\x85";
    inline static const std::string tooltip = "Load image";
    inline static constexpr uint8_t order = 1;

    virtual uint8_t GetMenuOrder() override
    {
        return order;
    }

    virtual void Serialize(std::ostream& out) override;
    virtual void Deserialize(std::istream& in) override;
};

ACMB_GUI_NAMESPACE_END
