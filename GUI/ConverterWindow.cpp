#include "ConverterWindow.h"
#include "MainWindow.h"
#include "Serializer.h"
#include "./../Transforms/converter.h"

ACMB_GUI_NAMESPACE_BEGIN

ConverterWindow::ConverterWindow( const Point& gridPos )
: PipelineElementWindow( "Converter", gridPos, PEFlags_StrictlyOneInput | PEFlags_StrictlyOneOutput )
{

}

void ConverterWindow::DrawPipelineElementControls()
{
    ImGui::Text( "Convert to Pixel Format:" );
    ImGui::RadioButton( "Gray8", ( int* ) &_dstPixelFormat, int( PixelFormat::Gray8 ) );
    ImGui::RadioButton( "Gray16", ( int* ) &_dstPixelFormat, int( PixelFormat::Gray16 ) );
    ImGui::RadioButton( "RGB24", ( int* ) &_dstPixelFormat, int( PixelFormat::RGB24 ) );
    ImGui::RadioButton( "RGB48", ( int* ) &_dstPixelFormat, int( PixelFormat::RGB48 ) );
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

void ConverterWindow::Deserialize(std::istream& in)
{
    PipelineElementWindow::Deserialize(in);
    _dstPixelFormat = acmb::gui::Deserialize<PixelFormat>( in, _remainingBytes );
}

int ConverterWindow::GetSerializedStringSize() const
{
    return PipelineElementWindow::GetSerializedStringSize() + gui::GetSerializedStringSize( _dstPixelFormat );
}

REGISTER_TOOLS_ITEM( ConverterWindow );

ACMB_GUI_NAMESPACE_END