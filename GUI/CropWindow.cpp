#include "CropWindow.h"
#include "MainWindow.h"
#include "Serializer.h"
#include "ImGuiHelpers.h"

#include "./../Transforms/ResizeTransform.h"

ACMB_GUI_NAMESPACE_BEGIN

CropWindow::CropWindow(  )
: PipelineElementWindow( "Crop" )
, SettingsInterpolationUser<CropTransform>( this, { 0, 0, 10000, 10000 })
{
}

void CropWindow::DrawPipelineElementControls()
{
    Size inputBitmapSize = { 65535, 65535 };
    if ( auto pPrimaryInput = GetInput(); pPrimaryInput && pPrimaryInput->GetPreviewedFrameNumber() >=0 )
        if ( auto inputBitmapSizeExp = pPrimaryInput->GetBitmapSize() )
            inputBitmapSize = inputBitmapSizeExp.value();

    UI::DragInt( "Left", &_dstRect.x, 1.0f, 0, inputBitmapSize.width - _dstRect.width - 1, "Left boundary of the crop area", nullptr );
    UI::DragInt( "Top", &_dstRect.y, 1.0f, 0, inputBitmapSize.height - _dstRect.height - 1,"Top boundary of the crop area", nullptr);
    UI::DragInt( "Width", &_dstRect.width, 1.0f, 1, inputBitmapSize.width - _dstRect.x,  "Width of the crop area", nullptr);
    UI::DragInt( "Height", &_dstRect.height, 1.0f, 1, inputBitmapSize.height - _dstRect.y, "Height of the crop area", nullptr);
}

void CropWindow::OnPreviewedFrameNumberChanged(int val)
{
    PipelineElementWindow::OnPreviewedFrameNumberChanged(val);
    _dstRect = GetInterpolatedSettings(GetPreviewedFrameNumber());
}

void CropWindow::OnKeyframeCommited()
{
    InsertOrAssignSettings(GetPreviewedFrameNumber(), _dstRect );
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

Expected<IBitmapPtr, std::string> CropWindow::GeneratePreviewBitmap(bool forNextElement, bool fullSize)
{
    auto pInputBitmapOrErr = GetInputPreview(true, true);
    if ( !pInputBitmapOrErr )
        return unexpected(pInputBitmapOrErr.error());

    if ( forNextElement )
    {
        auto pInputBitmap = pInputBitmapOrErr.value();

        const Size inputPreviewSize{ int(pInputBitmap->GetWidth()), int(pInputBitmap->GetHeight()) };
        const auto inputSizeExp = GetInput()->GetBitmapSize();
        if ( !inputSizeExp )
            return unexpected(inputSizeExp.error());

        const Size inputSize = inputSizeExp.value();
        const float xFactor = float(inputPreviewSize.width) / float(inputSize.width);
        const float yFactor = float(inputPreviewSize.height) / float(inputSize.height);

        const Rect cropArea
        {
            .x = std::clamp(int(_dstRect.x * xFactor), 0, inputPreviewSize.width - 1),
            .y = std::clamp(int(_dstRect.y * yFactor), 0, inputPreviewSize.height - 1),
            .width = std::clamp(int(_dstRect.width * xFactor), 1, inputPreviewSize.width - cropArea.x),
            .height = std::clamp(int(_dstRect.height * yFactor), 1, inputPreviewSize.height - cropArea.y)
        };

        auto pCroppedBitmap = CropTransform::Crop(pInputBitmap, cropArea);
        const Size finalSize = ResizeTransform::GetSizeWithPreservedRatio(Size{ cropArea.width, cropArea.height }, inputPreviewSize);
        return ResizeTransform::Resize(pCroppedBitmap, finalSize);
    }

    return pInputBitmapOrErr.value()->Clone();
}

Expected<Size, std::string> CropWindow::GetBitmapSize()
{
    return Size{ _dstRect.width, _dstRect.height };
}

void CropWindow::DrawOnPreviewImage(ImDrawList* pDrawList, ImVec2 topLeftPos, ImVec2 previewSize)
{
    auto origSize = GetInput()->GetBitmapSize();
    if ( !origSize )
        return;

    ImVec2 scale = 
    {
        previewSize.x / float( origSize->width ),
        previewSize.y / float( origSize->height )
    };

    RectF rectToDraw
    {
        .x = topLeftPos.x + _dstRect.x * scale.x,
        .y = topLeftPos.y + _dstRect.y * scale.y,
        .width = _dstRect.width * scale.x,
        .height = _dstRect.height * scale.y
    };

    pDrawList->AddRect( 
        ImVec2{ rectToDraw.x, rectToDraw.y },
        ImVec2{ rectToDraw.x + rectToDraw.width, rectToDraw.y + rectToDraw.height },
        IM_COL32( 255, 0, 0, 255 ),
        0.0f,
        ImDrawFlags_RoundCornersNone,
        3.0f
    );
}

REGISTER_TOOLS_ITEM( CropWindow );

ACMB_GUI_NAMESPACE_END
