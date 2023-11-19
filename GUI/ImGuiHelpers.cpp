#include "ImGuiHelpers.h"
#include "window.h"
#include "FontRegistry.h"

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

    static constexpr float cModalWindowWidth = 400.0f;
    static constexpr float cModalWindowPaddingX = 12.0f;
    static constexpr float cModalWindowPaddingY = 16.0f;
    static constexpr float cDefaultItemSpacing = 8.0f;
    static constexpr float cButtonPadding = 8.0f;

    void ShowModalMessage( const std::vector<std::string>& msgs, ModalMessageType msgType, bool& isEnabled )
    {
        ImGui::PushStyleColor( ImGuiCol_ModalWindowDimBg, ImVec4( 1, 0.125f, 0.125f, ImGui::GetStyle().Colors[ImGuiCol_ModalWindowDimBg].w ) );

        std::string title;
        if ( msgType == ModalMessageType::Error )
            title = "Error";
        else if ( msgType == ModalMessageType::Success )
            title = "Success";
        else //if ( modalMessageType_ == ModalMessageType::Help )
            title = "Help";

        const std::string titleImGui = " " + title + "##modal";

        if ( !ImGui::IsPopupOpen( " Error##modal" ) && !ImGui::IsPopupOpen( " Success##modal" ) && !ImGui::IsPopupOpen( " Help##modal" ) )
        {
            ImGui::OpenPopup( titleImGui.c_str() );
        }

        const auto menuScaling = acmb::gui::Window::cMenuScaling;

        const ImVec2 errorWindowSize{ cModalWindowWidth * menuScaling, -1 };
        ImGui::SetNextWindowSize( errorWindowSize, ImGuiCond_Always );
        ImGui::PushStyleVar( ImGuiStyleVar_WindowPadding, { cModalWindowPaddingX * menuScaling, cModalWindowPaddingY * menuScaling } );
        ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, { 2.0f * cDefaultItemSpacing * menuScaling, 3.0f * cDefaultItemSpacing * menuScaling } );
        if ( ImGui::BeginPopupModal( titleImGui.c_str(), nullptr,
                                           ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar ) )
        {
            auto headerFont = acmb::gui::FontRegistry::Instance().bigBold;
            if ( headerFont )
                ImGui::PushFont( headerFont );

            const auto headerWidth = ImGui::CalcTextSize( title.c_str() ).x;

            ImGui::SetCursorPosX( (errorWindowSize.x - headerWidth) * 0.5f );
            ImGui::Text( "%s", title.c_str() );

            if ( headerFont )
                ImGui::PopFont();

            auto font = acmb::gui::FontRegistry::Instance().big;
            if ( font )
                ImGui::PushFont( font );

            for ( const auto& msg : msgs )
            {
                const float textWidth = ImGui::CalcTextSize( msg.c_str() ).x;

                if ( textWidth < errorWindowSize.x )
                {
                    ImGui::SetCursorPosX( (errorWindowSize.x - textWidth) * 0.5f );
                    ImGui::Text( "%s", msg.c_str() );
                }
                else
                {
                    ImGui::TextWrapped( "%s", msg.c_str() );
                }
            }

            const auto style = ImGui::GetStyle();
            ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { style.FramePadding.x, cButtonPadding * menuScaling } );
            if ( ImGui::Button( "Okay", { -1, 0 } ) || ImGui::IsKeyPressed( ImGuiKey_Enter ) ||
                 (ImGui::IsMouseClicked( 0 ) && !(ImGui::IsAnyItemHovered() || ImGui::IsWindowHovered( ImGuiHoveredFlags_AnyWindow ))) )
            {
                isEnabled = false;
                ImGui::CloseCurrentPopup();
            }

            if ( font )
                ImGui::PopFont();

            ImGui::PopStyleVar();
            ImGui::EndPopup();
        }

        ImGui::PopStyleVar( 2 );
        ImGui::PopStyleColor();
    }
}