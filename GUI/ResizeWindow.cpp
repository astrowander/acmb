#include "ResizeWindow.h"
#include "MainWindow.h"
#include "Serializer.h"
#include "ImGuiHelpers.h"
#include "./../Transforms/ResizeTransform.h"
#include "./../Transforms/DebayerTransform.h"

ACMB_GUI_NAMESPACE_BEGIN

ResizeWindow::ResizeWindow(  )
    : PipelineElementWindow( "Resize" )
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
    if ( pSource->GetPixelFormat() == PixelFormat::Bayer16 )
    {
        pSource = DebayerTransform::Debayer(pSource, pSource->GetCameraSettings());
    }
    return ResizeTransform::Resize( pSource, _dstSize );
}

void ResizeWindow::Serialize( std::ostream& out ) const
{
    PipelineElementWindow::Serialize( out );
    gui::Serialize( _dstSize, out );
}

bool ResizeWindow::Deserialize( std::istream& in )
{
    if ( !PipelineElementWindow::Deserialize( in ) ) return false;
    _dstSize = gui::Deserialize<Size>( in, _remainingBytes );
    return true;
}

int ResizeWindow::GetSerializedStringSize() const
{
    return PipelineElementWindow::GetSerializedStringSize() + gui::GetSerializedStringSize( _dstSize );
}

Expected<IBitmapPtr, std::string> ResizeWindow::GeneratePreviewBitmap(bool forNextElement, bool fullSize)
{
    if ( !GetInput() )
        return unexpected("Primary input of the '" + _name + "' element is not set");

    auto pInputBitmapOrErr = GetInputPreview(true, fullSize);
    if ( !pInputBitmapOrErr )
        return unexpected(pInputBitmapOrErr.error());

    auto pInputBitmap = pInputBitmapOrErr.value();
    const Size inputPreviewSize{ int( pInputBitmap->GetWidth() ), int( pInputBitmap->GetHeight() ) };
    const Size inputSize = GetInput()->GetBitmapSize().value_or(inputPreviewSize);
    const Size dstSize = fullSize ? inputSize : _dstSize;
    const Size regionAvail{ int(MainWindow::GetInstance().GetImageRegionAvail().width), int(MainWindow::GetInstance().GetImageRegionAvail().height) };

    if ( dstSize.width <= regionAvail.width && dstSize.height <= regionAvail.height )
    {
        return ResizeTransform::Resize( pInputBitmap, dstSize );
    }

    Size adjustedSize;
    
    const double regionAspect = double(regionAvail.width) / double(regionAvail.height);
    const double dstAspect = double(dstSize.width) / double(dstSize.height);

    if ( dstAspect > regionAspect )
    {
        adjustedSize.width = regionAvail.width;
        adjustedSize.height = int( double(regionAvail.width) / dstAspect );
    }
    else
    {
        adjustedSize.height = regionAvail.height;
        adjustedSize.width = int( double(regionAvail.height) * dstAspect );
    }

    return ResizeTransform::Resize(pInputBitmap, adjustedSize);
}

Expected<Size, std::string> ResizeWindow::GetBitmapSize()
{
    return _dstSize;
}

REGISTER_TOOLS_ITEM( ResizeWindow )

ACMB_GUI_NAMESPACE_END
