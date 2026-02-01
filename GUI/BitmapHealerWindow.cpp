#include "BitmapHealerWindow.h"
#include "Serializer.h"
#include "MainWindow.h"
#include "ImGuiHelpers.h"

ACMB_GUI_NAMESPACE_BEGIN

BitmapHealerWindow::BitmapHealerWindow()
: PipelineElementWindow( "Bitmap Healer" )
{
    _patches.emplace_back();
    _currentPatch = 0;
}

void BitmapHealerWindow::DrawPipelineElementControls()
{
    ImGui::PushItemWidth( _itemWidth );

    ImGui::Text( "Copy From Coords" );

    auto& currentPatch = _patches[_currentPatch];
    currentPatch.gamma = 1.0f;
    //ImGui::SameLine();
    UI::DragInt2( "##copyfrom", (int*) &currentPatch.from, 1.0f, 0, 65535, "Coordinates of the copy source", this);
    ImGui::Text( "Paste To Coords" );
    UI::DragInt2( "##pasteto", (int*) &currentPatch.to, 1.0f, 0, 65535, "Coordinates of the copy destination", this);
    ImGui::Text( "Radius of Patch" );
    UI::DragInt( "##radius", &currentPatch.radius, 1.0f, 1, 65535, "Radius of the copy", this );
    ImGui::Text( "Index of Patch" );
    UI::InputInt( "##indexofpatch", &_currentPatch, 1, 0, 0, int( _patches.size() ), "Patch number", this );
    if ( _currentPatch == _patches.size() )
        _patches.emplace_back();

    ImGui::PopItemWidth();
}

Expected<IBitmapPtr, std::string> BitmapHealerWindow::GeneratePreviewBitmap(bool, bool fullSize)
{
    auto pInputBitmapOrErr = GetInputPreview(true, fullSize);
    if ( !pInputBitmapOrErr )
        return unexpected( pInputBitmapOrErr.error() );

    auto pInputBitmap = pInputBitmapOrErr.value();
    auto bitmapSize = GetInput()->GetBitmapSize().value_or(Size{ int(pInputBitmap->GetWidth()), int(pInputBitmap->GetHeight()) });
    const float scale = pInputBitmap->GetWidth() / float( bitmapSize.width );
    
    std::vector<BitmapHealer::Patch> patches = _patches;
    for ( auto& patch : patches )
    {
        patch.from.x = int( patch.from.x * scale + 0.5f );
        patch.from.y = int( patch.from.y * scale + 0.5f );
        patch.to.x = int( patch.to.x * scale + 0.5f );
        patch.to.y = int( patch.to.y * scale + 0.5f );
        patch.radius = int( patch.radius * scale + 0.5f );
    }

    return BitmapHealer::ApplyTransform( pInputBitmap, patches );
}

IBitmapPtr BitmapHealerWindow::ProcessBitmapFromPrimaryInput( IBitmapPtr pSource, size_t )
{
    return BitmapHealer::ApplyTransform( pSource, _patches );
}

void BitmapHealerWindow::Serialize( std::ostream& out ) const
{
    PipelineElementWindow::Serialize( out );
    gui::Serialize( _patches, out );
    gui::Serialize( _currentPatch, out );
}

bool BitmapHealerWindow::Deserialize( std::istream& in )
{
    if ( !PipelineElementWindow::Deserialize( in ) ) return false;
    _patches = gui::Deserialize<BitmapHealer::Settings>( in, _remainingBytes );
    _currentPatch = gui::Deserialize<int>( in, _remainingBytes );
    return true;
}

int BitmapHealerWindow::GetSerializedStringSize() const
{
    return PipelineElementWindow::GetSerializedStringSize()
        + gui::GetSerializedStringSize( _patches )
        + gui::GetSerializedStringSize( _currentPatch );
}

REGISTER_TOOLS_ITEM( BitmapHealerWindow );

ACMB_GUI_NAMESPACE_END