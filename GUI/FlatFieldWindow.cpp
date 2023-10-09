#include "FlatFieldWindow.h"
#include "MainWindow.h"
#include "./../Transforms/BitmapDivisor.h"

ACMB_GUI_NAMESPACE_BEGIN

DivideImageWindow::DivideImageWindow( const Point& gridPos )
    : PipelineElementWindow( "Apply Flat Field", gridPos, PEFlags::PEFlags_StrictlyTwoInputs | PEFlags::PEFlags_StrictlyOneOutput )
{
}

void DivideImageWindow::DrawPipelineElementControls()
{
    ImGui::Text( "Dark Frame Is on:" );
    ImGui::RadioButton( "Top", &_flatFieldIsOnTop, 1 );
    ImGui::RadioButton( "Left", &_flatFieldIsOnTop, 0 );
    ImGui::DragFloat( "Intensity", &_intensity, 0.1f, 0.0f, 100.0f );
}

std::expected<IBitmapPtr, std::string> DivideImageWindow::RunTask( size_t i )
{
    if ( _completedTaskCount == 0 )
    {
        const auto pDarkFrameWindow = ( _flatFieldIsOnTop ) ? GetTopInput() : GetLeftInput();
        if ( !pDarkFrameWindow )
            return std::unexpected( "No dark frame" );

        const auto darkFrameRes = pDarkFrameWindow->RunTaskAndReportProgress( 0 );
        if ( !darkFrameRes )
            return darkFrameRes;

        _pFlatField = *darkFrameRes;
    }

    const auto pInput = ( _flatFieldIsOnTop ) ? GetLeftInput() : GetTopInput();
    if ( !pInput )
        return std::unexpected( "No input " );

    if ( _taskCount == 0 )
    {
        _taskCount = pInput->GetTaskCount();
    }

    const auto taskRes = pInput->RunTaskAndReportProgress( i );
    if ( !taskRes.has_value() )
        return std::unexpected( taskRes.error() );

    return BitmapDivisor::Divide( *taskRes, { .pDivisor = _pFlatField, .intensity = _intensity } );
}

REGISTER_TOOLS_ITEM( DivideImageWindow );

ACMB_GUI_NAMESPACE_END