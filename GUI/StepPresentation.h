#pragma once
#include "ScrollablePanels.h"

ACMB_GUI_NAMESPACE_BEGIN

inline void RenderCompactStepCard(const void* element, const char* name, ImVec2 size,
    float progress, const char* status)
{
    ImGui::PushID(element);
    if (ImGui::BeginChild("card", size, ImGuiChildFlags_Borders,
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse))
    {
        CenteredCardCaption(name);
        const auto padding = ImGui::GetStyle().WindowPadding;
        ImGui::SetCursorPos({padding.x, 24 + padding.y});
        ImGui::ProgressBar(std::clamp(progress, 0.0f, 1.0f), {ImGui::GetContentRegionAvail().x, 0});
        ImGui::TextUnformatted(status);
    }
    ImGui::EndChild();
    ImGui::PopID();
}

template<class DrawControls>
void RenderStepSettings(const void* element, const char* name, bool locked, DrawControls drawControls)
{
    ImGui::PushID(element);
    // Per-element children retain separate scroll/focus state in the inspector.
    const auto& style = ImGui::GetStyle();
    const float width = std::max(1.0f, ImGui::GetContentRegionAvail().x -
        2 * style.WindowPadding.x - style.ScrollbarSize);
    if (BeginVerticalPanel("settings", {0, 0}, width, false))
    {
        const std::string caption(name);
        ImGui::TextWrapped("%s", caption.substr(0, caption.find("##")).c_str());
        ImGui::Separator();
        const float available = std::max(1.0f, ImGui::GetContentRegionAvail().x);
        ImGui::BeginDisabled(locked);
        ImGui::PushItemWidth(available);
        drawControls(available);
        ImGui::PopItemWidth();
        ImGui::EndDisabled();
    }
    ImGui::EndChild();
    ImGui::PopID();
}

inline void FitSettingWidth(const char* label)
{
    const float labelWidth = ImGui::CalcTextSize(label, nullptr, true).x;
    const float reserved = labelWidth > 0 ? labelWidth + ImGui::GetStyle().ItemInnerSpacing.x : 0;
    ImGui::SetNextItemWidth(std::min(ImGui::CalcItemWidth(),
        std::max(1.0f, ImGui::GetContentRegionAvail().x - reserved)));
}

ACMB_GUI_NAMESPACE_END
