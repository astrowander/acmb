#include "SaturationWindow.h"
#include "Serializer.h"
#include "MainWindow.h"
#include "ImGuiHelpers.h"

ACMB_GUI_NAMESPACE_BEGIN

SaturationWindow::SaturationWindow(  )
: PipelineElementWindow( "Saturation" )
, SettingsInterpolationUser<SaturationTransform>(this, SaturationTransform::Settings{})
{
}

void SaturationWindow::DrawPipelineElementControls()
{
    UI::DragFloat( "Saturation", &_saturationSettings, 0.01f, 0.0f, 4.0f, "Saturation factor", this );
}

Expected<void, std::string> SaturationWindow::GeneratePreviewBitmap()
{
    auto pInputBitmapOrErr = GetInput()->GetPreviewBitmap();
    if ( !pInputBitmapOrErr )
        return unexpected(pInputBitmapOrErr.error());

    auto pInputBitmap = pInputBitmapOrErr.value()->Clone();
    _pPreviewBitmap = SaturationTransform::Saturate( pInputBitmap, _saturationSettings );
    return {};
}

IBitmapPtr SaturationWindow::ProcessBitmapFromPrimaryInput( IBitmapPtr pSource, size_t frameIndex)
{
    auto interpolatedSettings = GetInterpolatedSettings(int(frameIndex));
    return SaturationTransform::Saturate( pSource, interpolatedSettings);
}

void SaturationWindow::OnPreviewedFrameNumberChanged(int val)
{
    PipelineElementWindow::OnPreviewedFrameNumberChanged(val);
    _saturationSettings = GetInterpolatedSettings(GetPreviewedFrameNumber());
}

void SaturationWindow::OnKeyframeCommited()
{
    InsertOrAssignSettings(GetPreviewedFrameNumber(), _saturationSettings);
}

void SaturationWindow::Serialize( std::ostream& out ) const
{
    PipelineElementWindow::Serialize( out );
    gui::Serialize( _saturationSettings, out );
    SettingsInterpolationUser<SaturationTransform>::Serialize(out);
}

bool SaturationWindow::Deserialize( std::istream& in )
{
    if ( !PipelineElementWindow::Deserialize( in ) ) return false;
    _saturationSettings = gui::Deserialize<float>( in, _remainingBytes );
    SettingsInterpolationUser<SaturationTransform>::Deserialize(in, _remainingBytes);
    return true;
}

int SaturationWindow::GetSerializedStringSize() const
{
    return PipelineElementWindow::GetSerializedStringSize() 
    + gui::GetSerializedStringSize( _saturationSettings )
    + SettingsInterpolationUser<SaturationTransform>::GetSerializedStringSize();
}

REGISTER_TOOLS_ITEM( SaturationWindow );

ACMB_GUI_NAMESPACE_END
