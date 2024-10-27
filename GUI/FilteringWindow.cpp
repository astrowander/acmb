#include "FilteringWindow.h"
#include "./../Registrator/SortingHat.h"

ACMB_GUI_NAMESPACE_BEGIN

FilteringWindow::FilteringWindow( const Point& gridPos )
    : PipelineElementWindow( "Sort Frames", gridPos, PEFlags_StrictlyOneInput | PEFlags_StrictlyOneOutput )
{
    _taskCount = 1;
}

void FilteringWindow::DrawPipelineElementControls()
{
    ImGui::Combo( "Filter Mode", (int*)& _filterMode, "Frame Count\0Percentage\0QualityThreshold\0");
    switch ( _filterMode )
    {
        case FilterMode::FrameCount:
            ImGui::DragInt( "Frame Count", &_frameCount, 1.0f, 1, 65535 );
            break;
        case FilterMode::Percentage:
            ImGui::DragFloat( "Percentage", &_percentage, 0.001f, 0.0f, 1.0f );
            break;
        case FilterMode::QualityThreshold:
            ImGui::DragFloat( "Threshold", &_qualityThreshold, 0.001f, 0.0f, 1.0f );
            break;
    }
}

Expected<void, std::string> FilteringWindow::SortFrames()
{
    auto pInput = GetPrimaryInput();
    if ( !pInput )
        return unexpected( "No primary input for the'" + _name + "' element" );

    const auto taskCount = pInput->GetTaskCount();
    if ( taskCount == 0 )
        return unexpected( "No input frames for the'" + _name + "' element" );

    auto bitmapOrError = pInput->RunTaskAndReportProgress( 0 );
    if ( !bitmapOrError )
        return unexpected( bitmapOrError.error() );

    SortingHat sortingHat( **bitmapOrError );
    sortingHat.AddFrame( *bitmapOrError );
    _taskReadiness = 1.0f / taskCount;

    for ( size_t i = 1; i < taskCount; ++i )
    {
        bitmapOrError = pInput->RunTaskAndReportProgress( i );
        if ( !bitmapOrError )
            return unexpected( bitmapOrError.error() );

        sortingHat.AddFrame( *bitmapOrError );
        _taskReadiness += 1.0f / taskCount;
    }

    _completedTaskCount = 1;
    _taskReadiness = 0;
}

Expected<IBitmapPtr, std::string> FilteringWindow::RunTask( size_t i )
{
    if ( i == 0 )
        SortFrames();

    return _sortedFrames[i];
}

ACMB_GUI_NAMESPACE_END