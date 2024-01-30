#include "ImGuiHelpers.h"
#include "window.h"
#include "FontRegistry.h"
#include "MainWindow.h"

using namespace acmb::gui;

namespace UI
{
    void SetTooltipIfHovered( const std::string& text, float scaling )
    {
        const bool isInterfaceLocked = MainWindow::GetInstance( FontRegistry::Instance() ).IsInterfaceLocked();

        if ( !isInterfaceLocked && ( !ImGui::IsItemHovered() || ImGui::IsItemActive() ) )
            return;

        if ( isInterfaceLocked && ( !ImGui::IsMouseDown( ImGuiMouseButton_Left ) || !ImGui::IsItemHovered() ) )
            return;

        assert( scaling > 0.f );
        ImGui::PushFont( FontRegistry::Instance().byDefault );

        constexpr float cMaxWidth = 400.f;
        const auto& style = ImGui::GetStyle();
        auto textSize = ImGui::CalcTextSize( text.c_str(), nullptr, false, cMaxWidth * scaling - style.WindowPadding.x * 2 );
        ImGui::SetNextWindowSize( ImVec2{ textSize.x + style.WindowPadding.x * 2, 0 } );

        ImGui::BeginTooltip();

        if ( isInterfaceLocked )
            ImGui::PushStyleColor( ImGuiCol_Text, { 1.0f, 0.0f , 0.0f , 1.0f } );

        ImGui::TextWrapped( "%s", isInterfaceLocked ? "All controls are locked because computations are going on or another window is opened" : text.c_str());

        ImGui::PopFont();
        if ( isInterfaceLocked )
            ImGui::PopStyleColor();

        ImGui::EndTooltip();
    }

    static constexpr float cModalWindowWidth = 400.0f;
    static constexpr float cModalWindowPaddingX = 12.0f;
    static constexpr float cModalWindowPaddingY = 16.0f;
    static constexpr float cDefaultItemSpacing = 8.0f;
    static constexpr float cButtonPadding = 8.0f;
    static constexpr ImGuiSliderFlags ImGuiSliderFlags_ReadOnly = 1 << 21;

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

        const auto menuScaling = Window::cMenuScaling;

        const ImVec2 errorWindowSize{ cModalWindowWidth * menuScaling, -1 };
        ImGui::SetNextWindowSize( errorWindowSize, ImGuiCond_Always );
        ImGui::PushStyleVar( ImGuiStyleVar_WindowPadding, { cModalWindowPaddingX * menuScaling, cModalWindowPaddingY * menuScaling } );
        ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, { 2.0f * cDefaultItemSpacing * menuScaling, 3.0f * cDefaultItemSpacing * menuScaling } );
        if ( ImGui::BeginPopupModal( titleImGui.c_str(), nullptr,
                                           ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar ) )
        {
            auto headerFont = FontRegistry::Instance().bigBold;
            if ( headerFont )
                ImGui::PushFont( headerFont );

            const auto headerWidth = ImGui::CalcTextSize( title.c_str() ).x;

            ImGui::SetCursorPosX( (errorWindowSize.x - headerWidth) * 0.5f );
            ImGui::Text( "%s", title.c_str() );

            if ( headerFont )
                ImGui::PopFont();

            auto font = FontRegistry::Instance().big;
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

    void Button( const std::string& name, const ImVec2& size, std::function<void()> action, const std::string& tooltip, PipelineElementWindow* parent )
    {
        const bool isInterfaceLocked = MainWindow::GetInstance( FontRegistry::Instance() ).IsInterfaceLocked();
        if ( ImGui::Button( name.c_str(), size) && !isInterfaceLocked )
        {
            action();
            if ( parent )
                parent->ResetPreview();
        }

        SetTooltipIfHovered( tooltip, MainWindow::cMenuScaling );
    }

    void RadioButton( const std::string& label, int* v, int v_button, const std::string& tooltip, PipelineElementWindow* parent )
    {
        const bool pressed = ImGui::RadioButton( label.c_str(), *v == v_button) && !MainWindow::GetInstance(FontRegistry::Instance()).IsInterfaceLocked();
        if ( pressed )
        { 
            *v = v_button;
            if ( parent )
                parent->ResetPreview();
        }
        
        SetTooltipIfHovered( tooltip, MainWindow::cMenuScaling );
    }

    void Checkbox( const std::string& label, bool* v, const std::string& tooltip, PipelineElementWindow* parent )
    {
        const bool isInterfaceLocked = MainWindow::GetInstance( FontRegistry::Instance() ).IsInterfaceLocked();
        if ( ImGui::Checkbox( label.c_str(), v, isInterfaceLocked ) && parent )
        {
            parent->ResetPreview();
        }
        SetTooltipIfHovered( tooltip, MainWindow::cMenuScaling );
    }

    void DragInt( const std::string& label, int* v, float v_speed, int v_min, int v_max, const std::string& tooltip, PipelineElementWindow* parent )
    {
        const bool isInterfaceLocked = MainWindow::GetInstance( FontRegistry::Instance() ).IsInterfaceLocked();
        if ( ImGui::DragInt( label.c_str(), v, v_speed, v_min, v_max, "%d", isInterfaceLocked ? ImGuiSliderFlags_ReadOnly : ImGuiSliderFlags_AlwaysClamp ) && parent )
        {
            parent->ResetPreview();
        }
        SetTooltipIfHovered( tooltip, MainWindow::cMenuScaling );
    }

    void DragFloat( const std::string& label, float* v, float v_speed, float v_min, float v_max,  const std::string& tooltip, PipelineElementWindow* parent )
    {
        const bool isInterfaceLocked = MainWindow::GetInstance( FontRegistry::Instance() ).IsInterfaceLocked();
        if ( ImGui::DragFloat( label.c_str(), v, v_speed, v_min, v_max, "%.3f", isInterfaceLocked ? ImGuiSliderFlags_ReadOnly : ImGuiSliderFlags_AlwaysClamp ) && parent )
        {
            parent->ResetPreview();
        }
        SetTooltipIfHovered( tooltip, MainWindow::cMenuScaling );
    }

    static bool Items_SingleStringGetter( void* data, int idx, const char** out_text )
    {
        // FIXME-OPT: we could pre-compute the indices to fasten this. But only 1 active combo means the waste is limited.
        const char* items_separated_by_zeros = ( const char* ) data;
        int items_count = 0;
        const char* p = items_separated_by_zeros;
        while ( *p )
        {
            if ( idx == items_count )
                break;
            p += strlen( p ) + 1;
            items_count++;
        }
        if ( !*p )
            return false;
        if ( out_text )
            *out_text = p;
        return true;
    }

    void Combo( const std::string& label, int* current_item, const std::string& items_separated_by_zeros, const std::string& tooltip, PipelineElementWindow* parent )
    {
        const bool isInterfaceLocked = MainWindow::GetInstance( FontRegistry::Instance() ).IsInterfaceLocked();

        int items_count = 0;
        if ( !isInterfaceLocked )
        {
            const char* p = items_separated_by_zeros.c_str();
            while ( *p )
            {
                p += strlen( p ) + 1;
                items_count++;
            }
        }

        if ( ImGui::Combo( label.c_str(), current_item, Items_SingleStringGetter, ( void* ) items_separated_by_zeros.c_str(), items_count ) && parent )
        {
            parent->ResetPreview();
        }
        SetTooltipIfHovered( tooltip, MainWindow::cMenuScaling );
    }
}
