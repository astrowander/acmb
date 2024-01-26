#include "ResizeWindow.h"
#include "MainWindow.h"
#include "Serializer.h"
#include "ImGuiHelpers.h"
#include "./../Transforms/ResizeTransform.h"
ACMB_GUI_NAMESPACE_BEGIN

ResizeWindow::ResizeWindow( const Point& gridPos )
    : PipelineElementWindow( "Resize", gridPos, PEFlags_StrictlyOneInput | PEFlags_StrictlyOneOutput )
{
}

void ResizeWindow::DrawPipelineElementControls()
{
    ImGui::Text( "Destination Size" );
    UI::DragInt( "Width", &_dstSize.width, 1.0f, 2, 65535, "Width of the resized image", this );
    UI::DragInt( "Height", &_dstSize.height, 1.0f, 2, 65535, "Height of the resized image", this );
}

IBitmapPtr ResizeWindow::ProcessBitmapFromPrimaryInput( IBitmapPtr pSource, size_t )
{
    return ResizeTransform::Resize( pSource, _dstSize );
}

void ResizeWindow::Serialize( std::ostream& out ) const
{
    PipelineElementWindow::Serialize( out );
    gui::Serialize( _dstSize, out );
}

void ResizeWindow::Deserialize( std::istream& in )
{
    PipelineElementWindow::Deserialize( in );
    _dstSize = gui::Deserialize<Size>( in, _remainingBytes );
}

int ResizeWindow::GetSerializedStringSize() const
{
    return PipelineElementWindow::GetSerializedStringSize() + gui::GetSerializedStringSize( _dstSize );
}

Expected<void, std::string> ResizeWindow::GeneratePreviewBitmap()
{
    const auto pInputBitmap = GetPrimaryInput()->GetPreviewBitmap()->Clone();
    const Size inputPreviewSize{ int( pInputBitmap->GetWidth() ), int( pInputBitmap->GetHeight() ) };
    const auto inputSizeExp = GetBitmapSize();
    if ( !inputSizeExp )
        return unexpected( inputSizeExp.error() );

    const Size inputSize = inputSizeExp.value();
    const float dstAspectRatio = float( _dstSize.width ) / float( _dstSize.height );
    constexpr float pivotAspectRatio = 16.0f / 9.0f;

    Size previewSize;
    if ( dstAspectRatio > pivotAspectRatio )
    {
        previewSize.width = inputPreviewSize.width;
        previewSize.height = std::max( int( previewSize.width / dstAspectRatio ), 1 );
    }
    else
    {
        previewSize.height = inputPreviewSize.height;
        previewSize.width = std::max( int( previewSize.height * dstAspectRatio ), 1 );
    }

    _pPreviewBitmap = ResizeTransform::Resize( pInputBitmap, previewSize );
    return {};
}

Expected<Size, std::string> ResizeWindow::GetBitmapSize()
{
    return _dstSize;
}

REGISTER_TOOLS_ITEM( ResizeWindow )

ACMB_GUI_NAMESPACE_END