#include "CropWindow.h"
#include "MainWindow.h"
#include "Serializer.h"
#include "ImGuiHelpers.h"

#include "./../Transforms/ResizeTransform.h"

ACMB_GUI_NAMESPACE_BEGIN

CropWindow::CropWindow( const Point& gridPos )
: PipelineElementWindow( "Crop", gridPos, PEFlags_StrictlyOneInput | PEFlags_StrictlyOneOutput )
, SettingsInterpolationUser<CropTransform>( this, { 0, 0, 10000, 10000 })
{
}

void CropWindow::DrawPipelineElementControls()
{
    Size inputBitmapSize = { 65535, 65535 };
    if ( auto pPrimaryInput = GetPrimaryInput(); pPrimaryInput && pPrimaryInput->GetPreviewedFrameNumber() >=0 )
        if ( auto inputBitmapSizeExp = pPrimaryInput->GetBitmapSize() )
            inputBitmapSize = inputBitmapSizeExp.value();

    _dstRect.x = std::clamp(_dstRect.x, 0, inputBitmapSize.width - 1);
    _dstRect.y = std::clamp( _dstRect.y, 0, inputBitmapSize.height - 1 );
    _dstRect.width = std::clamp( _dstRect.width, 1, inputBitmapSize.width - _dstRect.x );
    _dstRect.height = std::clamp( _dstRect.height, 1, inputBitmapSize.height - _dstRect.y );

    UI::DragInt( "Left", &_dstRect.x, 1.0f, 0, inputBitmapSize.width - 1, "Left boundary of the crop area", this );
    UI::DragInt( "Top", &_dstRect.y, 1.0f, 0, inputBitmapSize.height - 1,"Top boundary of the crop area", this );
    UI::DragInt( "Width", &_dstRect.width, 1.0f, 1, inputBitmapSize.width - _dstRect.x,  "Width of the crop area", this );
    UI::DragInt( "Height", &_dstRect.height, 1.0f, 1, inputBitmapSize.height - _dstRect.y, "Height of the crop area", this );

    DrawFrameCounter();
}

void CropWindow::OnPreviewedFrameNumberChanged(int val)
{
    PipelineElementWindow::OnPreviewedFrameNumberChanged(val);
    _dstRect = GetInterpolatedSettings(_previewedFrameNumber);
}

void CropWindow::OnKeyframeCommited()
{
    AddSettings(_previewedFrameNumber, _dstRect );
}

IBitmapPtr CropWindow::ProcessBitmapFromPrimaryInput( IBitmapPtr pSource, size_t frameIndex )
{
    auto interpolatedSettings = GetInterpolatedSettings( int( frameIndex ) );
    return CropTransform::Crop( pSource, interpolatedSettings);
}

void CropWindow::Serialize( std::ostream& out ) const
{
    PipelineElementWindow::Serialize( out );
    gui::Serialize( _dstRect, out );
    SettingsInterpolationUser<CropTransform>::Serialize( out );
}

bool CropWindow::Deserialize( std::istream& in )
{
    if ( !PipelineElementWindow::Deserialize( in ) ) return false;
    _dstRect = gui::Deserialize<Rect>( in, _remainingBytes );
    SettingsInterpolationUser<CropTransform>::Deserialize( in, _remainingBytes );
    return true;
}

int CropWindow::GetSerializedStringSize() const
{
    return PipelineElementWindow::GetSerializedStringSize() 
            + gui::GetSerializedStringSize( _dstRect ) 
            + SettingsInterpolationUser<CropTransform>::GetSerializedStringSize();
}

Expected<void, std::string> CropWindow::GeneratePreviewBitmap()
{
    auto pInputBitmapOrErr = GetPrimaryInput()->GetPreviewBitmap();
    if ( !pInputBitmapOrErr )
        return unexpected(pInputBitmapOrErr.error());

    auto pInputBitmap = pInputBitmapOrErr.value()->Clone();

    const Size inputPreviewSize{ int( pInputBitmap->GetWidth() ), int( pInputBitmap->GetHeight() ) };
    const auto inputSizeExp = GetPrimaryInput()->GetBitmapSize();
    if ( !inputSizeExp )
        return unexpected( inputSizeExp.error() );

    const Size inputSize = inputSizeExp.value();
    const float xFactor = float(inputPreviewSize.width) / float( inputSize.width );
    const float yFactor = float(inputPreviewSize.height) / float( inputSize.height );

    const Rect cropArea
    { 
        .x = std::clamp( int(_dstRect.x * xFactor ), 0, inputPreviewSize.width - 1 ),
        .y = std::clamp( int(_dstRect.y * yFactor ), 0, inputPreviewSize.height - 1 ),
        .width = std::clamp( int(_dstRect.width * xFactor ), 1, inputPreviewSize.width - cropArea.x ),
        .height = std::clamp( int(_dstRect.height * yFactor ), 1, inputPreviewSize.height - cropArea.y )
    };

    _pPreviewBitmap = CropTransform::Crop(pInputBitmap, cropArea );
    const Size finalSize = ResizeTransform::GetSizeWithPreservedRatio(Size{ cropArea.width, cropArea.height }, inputPreviewSize);
    _pPreviewBitmap = ResizeTransform::Resize(_pPreviewBitmap, finalSize);

    return {};
}

Expected<Size, std::string> CropWindow::GetBitmapSize()
{
    return Size{ _dstRect.width, _dstRect.height };
}

REGISTER_TOOLS_ITEM( CropWindow );

ACMB_GUI_NAMESPACE_END
