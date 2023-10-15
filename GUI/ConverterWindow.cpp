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
    ImGui::Text( "Pixel Format" );
    ImGui::RadioButton( "Gray8", ( int* ) &_dstPixelFormat, int( PixelFormat::Gray8 ) );
    ImGui::RadioButton( "Gray16", ( int* ) &_dstPixelFormat, int( PixelFormat::Gray16 ) );
    ImGui::RadioButton( "RGB24", ( int* ) &_dstPixelFormat, int( PixelFormat::RGB24 ) );
    ImGui::RadioButton( "RGB48", ( int* ) &_dstPixelFormat, int( PixelFormat::RGB48 ) );
}

std::expected<IBitmapPtr, std::string> ConverterWindow::RunTask( size_t i )
{
    try
    {
        auto pInput = GetLeftInput();
        if ( !pInput )
            pInput = GetTopInput();

        if ( !pInput )
            return std::unexpected( "No input element" );

        if ( _taskCount == 0 )
        {
            _taskCount = pInput->GetTaskCount();
        }

        const auto taskRes = pInput->RunTaskAndReportProgress( i );
        if ( !taskRes.has_value() )
            return std::unexpected( taskRes.error() );

        return Converter::Convert( taskRes.value(), _dstPixelFormat );
    }
    catch ( std::exception& e )
    {
        return std::unexpected( e.what() );
    }
}

void ConverterWindow::Serialize(std::ostream& out)
{
    PipelineElementWindow::Serialize(out);
    acmb::gui::Serialize(_dstPixelFormat, out);
}

void ConverterWindow::Deserialize(std::istream& in)
{
    PipelineElementWindow::Deserialize(in);
    _dstPixelFormat = acmb::gui::Deserialize<PixelFormat>( in );
}

REGISTER_TOOLS_ITEM( ConverterWindow );

ACMB_GUI_NAMESPACE_END