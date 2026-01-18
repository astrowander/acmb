#include "DarkFrameWindow.h"
#include "MainWindow.h"
#include "Serializer.h"

#include "./../Codecs/imagedecoder.h"
#include "../Transforms/BitmapSubtractor.h"
#include "../Transforms/HistogramBuilder.h"

#include "ImGuiHelpers.h"

ACMB_GUI_NAMESPACE_BEGIN

DarkFrameWindow::DarkFrameWindow( const Point& gridPos )
: PipelineElementWindow( "Subtract Dark Frame", gridPos, PEFlags::PEFlags_StrictlyOneInput | PEFlags::PEFlags_StrictlyOneOutput )
, FileListUser(this)
{
}

void DarkFrameWindow::DrawPipelineElementControls()
{
    UI::DragFloat( "Multiplier", &_multiplier, 0.001f, 0.2f, 5.0f, "Each pixel of the dark frame will be multiplied by this factor before subtracting", this );
    UI::Button( "Adjust Multiplier", { -1, 0 }, [&]
    {
        auto& mainWindow = MainWindow::GetInstance( FontRegistry::Instance() );
        mainWindow.LockInterface();
        IBitmapPtr pLightFrame = GetPrimaryInput()->RunTaskAndReportProgress(0).value_or( nullptr );
        _multiplier = BitmapSubtractor::AutoAdjustMultiplier(pLightFrame, FileListUser::ReadFrame(0).value_or(nullptr));
        mainWindow.UnlockInterface();
    }, "Calculate appropriate multiplier automatically", this );
    
    ImGui::Separator();
    if ( ImGui::CollapsingHeader( "List of Dark Frames" ) )
    {
        FileListUser::DrawControls();
    }
}

IBitmapPtr DarkFrameWindow::ProcessBitmapFromPrimaryInput( IBitmapPtr pSource, size_t )
{
    return BitmapSubtractor::Subtract( pSource, { .pBitmapToSubtract = FileListUser::GetBitmapOfStackedFrames(), .multiplier = _multiplier});
}

void DarkFrameWindow::Serialize( std::ostream& out ) const
{
    PipelineElementWindow::Serialize( out );
    FileListUser::Serialize( out );
    gui::Serialize( _multiplier, out );
}

std::string DarkFrameWindow::GetWindowName() const
{
    return _name;
}

std::string DarkFrameWindow::GetFileFilters() const
{
    return ImageDecoder::GetFilters();
}

int DarkFrameWindow::GetSerializedStringSize() const
{
    return PipelineElementWindow::GetSerializedStringSize() + FileListUser::GetSerializedStringSize() + gui::GetSerializedStringSize( _multiplier );
}

bool DarkFrameWindow::Deserialize( std::istream& in )
{
    if ( !PipelineElementWindow::Deserialize( in ) ) return false;
    if ( !FileListUser::Deserialize(in, _remainingBytes) ) return false;
    _multiplier = gui::Deserialize<float>( in, _remainingBytes );
    return true;
}

Expected<void, std::string> DarkFrameWindow::GeneratePreviewBitmap()
{
    auto pInputBitmapOrErr = GetPrimaryInput()->GetPreviewBitmap();
    if ( !pInputBitmapOrErr )
        return unexpected(pInputBitmapOrErr.error());

    auto pInputBitmap = pInputBitmapOrErr.value()->Clone();
    
    auto pDarkFrameOrErr = FileListUser::ReadFramePreview(0, { int( pInputBitmap->GetWidth() ), int( pInputBitmap->GetHeight() ) } );
    if ( !pDarkFrameOrErr )
        return unexpected(pDarkFrameOrErr.error());

    auto pDarkFrame = pDarkFrameOrErr.value()->Clone();

    _pPreviewBitmap = BitmapSubtractor::Subtract( pInputBitmap, {.pBitmapToSubtract = pDarkFrame, .multiplier = _multiplier});
    _pPreviewTexture = std::make_unique<Texture>( _pPreviewBitmap );
    return {};
}

REGISTER_TOOLS_ITEM( DarkFrameWindow )

ACMB_GUI_NAMESPACE_END