#pragma once
#include "ScrollablePanels.h"
#include <cmath>
#include <array>

ACMB_GUI_NAMESPACE_BEGIN

// RGB display values use 256 bins; alpha is intentionally excluded.
template<class GetScanline>
std::array<std::array<float, 256>, 3> BuildPreviewHistogram(uint32_t width, uint32_t height,
    GetScanline getScanline, uint32_t pixelStride = 3)
{
    std::array<std::array<float, 256>, 3> bins{};
    for (uint32_t y = 0; y < height; ++y)
    {
        const auto* row = getScanline(y);
        for (uint32_t x = 0; x < width; ++x)
            for (uint32_t channel = 0; channel < 3; ++channel)
                ++bins[channel][row[x * pixelStride + channel]];
    }
    return bins;
}

template<class Element>
float PipelineHeadProgress(Element* head)
{
    if (!head) return 0;
    const float progress = head->GetProgress();
    return std::isfinite(progress) ? std::clamp(progress, 0.0f, 1.0f) : 0;
}

inline void RenderCompactStepCard(const void* element, const char* name, ImVec2 size,
    const char* status)
{
    ImGui::PushID(element);
    if (ImGui::BeginChild("card", size, ImGuiChildFlags_Borders,
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse))
    {
        CenteredCardCaption(name);
        const auto padding = ImGui::GetStyle().WindowPadding;
        ImGui::SetCursorPos({padding.x, 24 + padding.y});
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
