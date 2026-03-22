#include "MedianBlurWindow.h"
#include "Serializer.h"
#include "MainWindow.h"
#include "ImGuiHelpers.h"

#include "./../Transforms/MedianBlurTransform.h"
#include "./../Transforms/DebayerTransform.h"

ACMB_GUI_NAMESPACE_BEGIN

MedianBlurWindow::MedianBlurWindow(  )
: PipelineElementWindow( "Median Blur" )
{
}

void MedianBlurWindow::DrawPipelineElementControls()
{
    UI::DragInt( "Radius", &_radius, 1.0f, 1, 100, "Radius of the median blur", this );
}

Expected<IBitmapPtr, std::string> MedianBlurWindow::GeneratePreviewBitmap(bool forNextElement, bool fullSize)
{
    auto pInputBitmapOrErr = GetInputPreview(true, fullSize);
    if ( !pInputBitmapOrErr )
        return unexpected(pInputBitmapOrErr.error());

    const Size previewSize = Size { int( (*pInputBitmapOrErr)->GetWidth() ), int( (*pInputBitmapOrErr)->GetHeight() ) };
    const Size originalSize = GetBitmapSize().value_or(previewSize);

    const float scaleFactor = std::min(float(previewSize.width) / originalSize.width, float(previewSize.height) / originalSize.height);
    const int scaledRadius = int(_radius * scaleFactor + 0.5f);

    auto pInputBitmap = pInputBitmapOrErr.value();
    return MedianBlurTransform::MedianBlur( pInputBitmap, 2 * scaledRadius + 1 );
}

IBitmapPtr MedianBlurWindow::ProcessBitmapFromPrimaryInput( IBitmapPtr pSource, size_t )
{
    if ( pSource->GetPixelFormat() == PixelFormat::Bayer16 )
        pSource = DebayerTransform::Debayer(pSource, pSource->GetCameraSettings());

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