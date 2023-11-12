#include "FlatFieldWindow.h"
#include "MainWindow.h"
#include "Serializer.h"
#include "./../Transforms/BitmapDivisor.h"

ACMB_GUI_NAMESPACE_BEGIN

DivideImageWindow::DivideImageWindow( const Point& gridPos )
    : PipelineElementWindow( "Apply Flat Field", gridPos, PEFlags::PEFlags_StrictlyTwoInputs | PEFlags::PEFlags_StrictlyOneOutput )
{
}

void DivideImageWindow::DrawPipelineElementControls()
{
    ImGui::Text( "Dark Frame Is on:" );
    ImGui::RadioButton( "Top", &_primaryInputIsOnLeft, 1 );
    ImGui::RadioButton( "Left", &_primaryInputIsOnLeft, 0 );
    ImGui::DragFloat( "Intensity", &_intensity, 0.1f, 0.0f, 100.0f );
}

IBitmapPtr DivideImageWindow::ProcessBitmapFromPrimaryInput( IBitmapPtr pSource, size_t )
{
    return BitmapDivisor::Divide( pSource, { .pDivisor = _pSecondaryInputResult, .intensity = _intensity } );
}

void DivideImageWindow::Serialize(std::ostream& out)
{
    PipelineElementWindow::Serialize(out);
    gui::Serialize(_intensity, out);
}

void DivideImageWindow::Deserialize(std::istream& in)
{
    PipelineElementWindow::Deserialize(in);
    _intensity = gui::Deserialize<float>(in);
}

REGISTER_TOOLS_ITEM( DivideImageWindow );

ACMB_GUI_NAMESPACE_END