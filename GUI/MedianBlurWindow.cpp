#include "MedianBlurWindow.h"
#include "Serializer.h"
#include "MainWindow.h"
#include "ImGuiHelpers.h"

#include "./../Transforms/MedianBlurTransform.h"

ACMB_GUI_NAMESPACE_BEGIN

MedianBlurWindow::MedianBlurWindow( const Point& gridPos )
: PipelineElementWindow( "Median Blur", gridPos, PEFlags_StrictlyOneInput | PEFlags_StrictlyOneOutput )
{
}

void MedianBlurWindow::DrawPipelineElementControls()
{
    UI::DragInt( "Radius", &_radius, 1.0f, 1, 100, "Radius of the median blur" );
}

Expected<void, std::string> MedianBlurWindow::GeneratePreviewBitmap()
{
    auto pInputBitmapOrErr = GetPrimaryInput()->GetPreviewBitmap();
    if ( !pInputBitmapOrErr )
        return unexpected(pInputBitmapOrErr.error());

    auto pInputBitmap = pInputBitmapOrErr.value()->Clone();
    _pPreviewBitmap = MedianBlurTransform::MedianBlur( pInputBitmap, 2 * _radius + 1 );
    return {};
}

IBitmapPtr MedianBlurWindow::ProcessBitmapFromPrimaryInput( IBitmapPtr pSource, size_t )
{
    return MedianBlurTransform::MedianBlur(pSource, 2 * _radius + 1 );
}

void MedianBlurWindow::Serialize( std::ostream& out ) const
{
    PipelineElementWindow::Serialize( out );
    gui::Serialize( _radius, out );
}

bool MedianBlurWindow::Deserialize( std::istream& in )
{
    if ( !PipelineElementWindow::Deserialize( in ) ) return false;
    _radius = gui::Deserialize<int>( in, _remainingBytes );
    return true;
}

int MedianBlurWindow::GetSerializedStringSize() const
{
    return PipelineElementWindow::GetSerializedStringSize()
        + gui::GetSerializedStringSize( _radius );
}

REGISTER_TOOLS_ITEM( MedianBlurWindow );

ACMB_GUI_NAMESPACE_END