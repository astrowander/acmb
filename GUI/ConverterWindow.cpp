#include "ConverterWindow.h"
#include "MainWindow.h"
#include "Serializer.h"
#include "ImGuiHelpers.h"
#include "./../Transforms/converter.h"

ACMB_GUI_NAMESPACE_BEGIN

ConverterWindow::ConverterWindow( const Point& gridPos )
: PipelineElementWindow( "Converter", gridPos, PEFlags_StrictlyOneInput | PEFlags_StrictlyOneOutput )
{

}

void ConverterWindow::DrawPipelineElementControls()
{
    ImGui::Text( "Convert to Pixel Format:" );
    UI::RadioButton( "Gray8", ( int* ) &_dstPixelFormat, int( PixelFormat::Gray8 ), "Grayscale, 8 bits per channel", this );
    UI::RadioButton( "Gray16", ( int* ) &_dstPixelFormat, int( PixelFormat::Gray16 ), "Grayscale, 16 bits per channel", this );
    UI::RadioButton( "RGB24", ( int* ) &_dstPixelFormat, int( PixelFormat::RGB24 ), "RGB colors, 8 bits per channel", this );
    UI::RadioButton( "RGB48", ( int* ) &_dstPixelFormat, int( PixelFormat::RGB48 ), "RGB colors, 16 bits per channel", this );
    UI::RadioButton( "YUV444", ( int* ) &_dstPixelFormat, int( PixelFormat::YUV24 ), "YUV colors, 8 bits per channel", this );
}

IBitmapPtr ConverterWindow::ProcessBitmapFromPrimaryInput( IBitmapPtr pSource, size_t )
{
    return Converter::Convert( pSource, _dstPixelFormat );
}

void ConverterWindow::Serialize(std::ostream& out) const
{
    PipelineElementWindow::Serialize(out);
    acmb::gui::Serialize(_dstPixelFormat, out);
}

bool ConverterWindow::Deserialize(std::istream& in)
{
    PipelineElementWindow::Deserialize(in);
    _dstPixelFormat = acmb::gui::Deserialize<PixelFormat>( in, _remainingBytes );
    return true;
}

int ConverterWindow::GetSerializedStringSize() const
{
    return PipelineElementWindow::GetSerializedStringSize() + gui::GetSerializedStringSize( _dstPixelFormat );
}

Expected<void, std::string> ConverterWindow::GeneratePreviewBitmap()
{
    auto pInputBitmapOrErr = GetPrimaryInput()->GetPreviewBitmap();
    if ( !pInputBitmapOrErr )
        return unexpected(pInputBitmapOrErr.error());

    auto pInputBitmap = pInputBitmapOrErr.value()->Clone();

    _pPreviewBitmap = Converter::Convert(pInputBitmap->Clone(), _dstPixelFormat);
    return {};
}

REGISTER_TOOLS_ITEM( ConverterWindow );

ACMB_GUI_NAMESPACE_END