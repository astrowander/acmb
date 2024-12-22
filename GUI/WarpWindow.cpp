#include "WarpWindow.h"
#include "Serializer.h"
#include "MainWindow.h"
#include "ImGuiHelpers.h"

ACMB_GUI_NAMESPACE_BEGIN

WarpWindow::WarpWindow( const Point& gridPos )
    : PipelineElementWindow( "Warp", gridPos, PEFlags_StrictlyOneInput | PEFlags_StrictlyOneOutput )
{
}

void WarpWindow::DrawPipelineElementControls()
{
    const auto drawControlsLine = [&] (int lineIndex)
    {
        ImGui::Text( "%3d", lineIndex );
        ImGui::SameLine();
        ImGui::PushItemWidth( 50.0f );
        const std::string xId = "##control_x_" + std::to_string( lineIndex );
        UI::DragFloat( xId, &_settings.controls[lineIndex].x, 0.01f, -2.0f, 2.0f, "Edit x coordinate of the control point", this );
        ImGui::SameLine();
        const std::string yId = "##control_y_" + std::to_string( lineIndex );
        UI::DragFloat( yId, &_settings.controls[lineIndex].y, 0.01f, -2.0f, 2.0f, "Edit y coordinate of the control point", this );
        ImGui::PopItemWidth();
    };
    
    if ( ImGui::CollapsingHeader( "Controls 0-3" ) )
    {
        for ( int i = 0; i < 4; ++i )
            drawControlsLine( i );
    }

    if ( ImGui::CollapsingHeader( "Controls 4-7" ) )
    {
        for ( int i = 4; i < 8; ++i )
            drawControlsLine( i );
    }

    if ( ImGui::CollapsingHeader( "Controls 8-11" ) )
    {
        for ( int i = 8; i < 12; ++i )
            drawControlsLine( i );
    }

    if ( ImGui::CollapsingHeader( "Controls 12-15" ) )
    {
        for ( int i = 12; i < 16; ++i )
            drawControlsLine( i );
    }
}

Expected<void, std::string> WarpWindow::GeneratePreviewBitmap()
{
    auto pInputBitmap = GetPrimaryInput()->GetPreviewBitmap()->Clone();
    _pPreviewBitmap = WarpTransform::Warp( pInputBitmap, _settings );
    return {};
}

IBitmapPtr WarpWindow::ProcessBitmapFromPrimaryInput( IBitmapPtr pSource, size_t )
{
    return WarpTransform::Warp( pSource, _settings );
}

void WarpWindow::Serialize( std::ostream& out ) const
{
    PipelineElementWindow::Serialize( out );
    gui::Serialize( _settings, out );
}

bool WarpWindow::Deserialize( std::istream& in )
{
    if ( !PipelineElementWindow::Deserialize( in ) ) return false;
    _settings = gui::Deserialize<WarpTransform::Settings>( in, _remainingBytes );
    return true;
}

int WarpWindow::GetSerializedStringSize() const
{
    return PipelineElementWindow::GetSerializedStringSize()
        + gui::GetSerializedStringSize( _settings );
}

REGISTER_TOOLS_ITEM( WarpWindow );

ACMB_GUI_NAMESPACE_END
