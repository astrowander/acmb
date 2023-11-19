#include "ImGuiHelpers.h"

namespace ImGui
{
    void SetTooltipIfHovered( const std::string& text, float scaling )
    {
        if ( !ImGui::IsItemHovered() || ImGui::IsItemActive() )
            return;

        assert( scaling > 0.f );

        constexpr float cMaxWidth = 400.f;
        const auto& style = ImGui::GetStyle();
        auto textSize = ImGui::CalcTextSize( text.c_str(), nullptr, false, cMaxWidth * scaling - style.WindowPadding.x * 2 );
        ImGui::SetNextWindowSize( ImVec2{ textSize.x + style.WindowPadding.x * 2, 0 } );

        ImGui::BeginTooltip();
        ImGui::TextWrapped( "%s", text.c_str() );
        ImGui::EndTooltip();
    }
}