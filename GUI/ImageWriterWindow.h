#pragma once
#include "PipelineElementWindow.h"
#include "Texture.h"

ACMB_NAMESPACE_BEGIN
class VideoEncoder;
ACMB_NAMESPACE_END

ACMB_GUI_NAMESPACE_BEGIN

class ImageWriterWindow : public PipelineElementWindow
{
    std::string _workingDirectory;
    std::string _fileName;
    std::string _extension;
    std::string _formatList;    
    int _formatId = 0;
    bool _keepOriginalFileName = false;

    int _frameRate = 25;
    int _quality = 3;

    std::shared_ptr<VideoEncoder> _pEncoder;
    std::unique_ptr<Texture> _pResultTexture;

    virtual IBitmapPtr ProcessBitmapFromPrimaryInput( IBitmapPtr pSource, size_t taskNumber ) override;
    virtual Expected<void, std::string> GeneratePreviewBitmap() override;
public:
    ImageWriterWindow( const Point& gridPos );
    virtual void DrawPipelineElementControls() override;
    virtual void Serialize( std::ostream& out ) const override;
    virtual void Deserialize( std::istream& in ) override;
    virtual int GetSerializedStringSize() const override;

    std::vector<std::string> ExportAllImages();

    SET_MENU_PARAMS( "\xef\x95\xad", "Export", "Choose a file or a directory where to save the results", 2 );
};

ACMB_GUI_NAMESPACE_END

