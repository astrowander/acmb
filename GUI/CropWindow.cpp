#include "CropWindow.h"
#include "MainWindow.h"
#include "Serializer.h"
#include "./../Transforms/CropTransform.h"

ACMB_GUI_NAMESPACE_BEGIN

CropWindow::CropWindow( const Point& gridPos )
: PipelineElementWindow( "Crop", gridPos, PEFlags_StrictlyOneInput | PEFlags_StrictlyOneOutput )
{
}

void CropWindow::DrawPipelineElementControls()
{
    ImGui::DragInt( "Left", &_dstRect.x, 1.0f, 1, 65535 );
    ImGui::DragInt( "Top", &_dstRect.y, 1.0f, 1, 65535 );
    ImGui::DragInt( "Width", &_dstRect.width, 1.0f, 1, 65535 );
    ImGui::DragInt( "Height", &_dstRect.height, 1.0f, 1, 65535 );    
}

std::expected<IBitmapPtr, std::string> CropWindow::RunTask( size_t i )
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

        return CropTransform::Crop( *taskRes, _dstRect );
    }
    catch ( std::exception& e )
    {
        return std::unexpected( e.what() );
    }
}

void CropWindow::Serialize(std::ostream& out)
{
    PipelineElementWindow::Serialize(out);
    gui::Serialize(_dstRect, out);
}

void CropWindow::Deserialize(std::istream& in)
{
    PipelineElementWindow::Deserialize(in);
    _dstRect = gui::Deserialize<Rect>(in);
}

REGISTER_TOOLS_ITEM( CropWindow );

ACMB_GUI_NAMESPACE_END
