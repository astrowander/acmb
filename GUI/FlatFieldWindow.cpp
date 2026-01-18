#include "FlatFieldWindow.h"
#include "MainWindow.h"
#include "Serializer.h"
#include "ImGuiHelpers.h"

#include "./../Codecs/imagedecoder.h"
#include "./../Transforms/BitmapDivisor.h"

ACMB_GUI_NAMESPACE_BEGIN

FlatFieldWindow::FlatFieldWindow( const Point& gridPos )
    : PipelineElementWindow( "Apply Flat Field", gridPos, PEFlags::PEFlags_StrictlyOneInput | PEFlags::PEFlags_StrictlyOneOutput )
    , FileListUser(this)
{
}

void FlatFieldWindow::DrawPipelineElementControls()
{
    UI::DragFloat( "Intensity", &_intensity, 0.1f, 0.0f, 500.0f, "The effect of the instrument can be weakened or enhanced. The default value is 100 percent", this );
    ImGui::Separator();
    if ( ImGui::CollapsingHeader( "List of Flat Frames" ) )
    {
        FileListUser::DrawControls();
    }
}

IBitmapPtr FlatFieldWindow::ProcessBitmapFromPrimaryInput( IBitmapPtr pSource, size_t )
{
    return BitmapDivisor::Divide( pSource, { .pDivisor = FileListUser::GetBitmapOfStackedFrames(), .intensity = _intensity } );
}

void FlatFieldWindow::Serialize(std::ostream& out) const
{
    PipelineElementWindow::Serialize(out);
    FileListUser::Serialize(out);
    gui::Serialize(_intensity, out);
}

bool FlatFieldWindow::Deserialize(std::istream& in)
{
    PipelineElementWindow::Deserialize(in);
    FileListUser::Deserialize(in, _remainingBytes);
    _intensity = gui::Deserialize<float>(in, _remainingBytes);
    return true;
}

int FlatFieldWindow::GetSerializedStringSize() const
{
    return PipelineElementWindow::GetSerializedStringSize() + gui::GetSerializedStringSize( _intensity ) + FileListUser::GetSerializedStringSize();
}

Expected<void, std::string> FlatFieldWindow::GeneratePreviewBitmap()
{
    auto pInputBitmapOrErr = GetPrimaryInput()->GetPreviewBitmap();
    if ( !pInputBitmapOrErr )
        return unexpected(pInputBitmapOrErr.error());

    auto pInputBitmap = pInputBitmapOrErr.value()->Clone();

    auto pSecondaryInputBitmapOrErr = FileListUser::ReadFramePreview(0, { int(pInputBitmap->GetWidth()), int(pInputBitmap->GetHeight()) });
    if ( !pSecondaryInputBitmapOrErr )
        return unexpected(pSecondaryInputBitmapOrErr.error());

    auto pSecondaryInputBitmap = pSecondaryInputBitmapOrErr.value()->Clone();

    _pPreviewBitmap = BitmapDivisor::Divide(pInputBitmap, { .pDivisor = pSecondaryInputBitmap, .intensity = _intensity});
    return {};
}

std::string FlatFieldWindow::GetWindowName() const
{
    return _name;
}

std::string FlatFieldWindow::GetFileFilters() const
{
    return ImageDecoder::GetFilters();
}

REGISTER_TOOLS_ITEM( FlatFieldWindow );

ACMB_GUI_NAMESPACE_END