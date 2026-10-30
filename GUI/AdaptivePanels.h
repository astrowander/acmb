#pragma once
#include "MainWindowLayout.h"

ACMB_GUI_NAMESPACE_BEGIN

struct SplitterDrag { float delta = 0; bool released = false; };
enum class LayoutSplitter { Toolbox, Inspector, Pipeline };

inline bool ResizeLayoutPanel(LayoutPreferences& preferences, const MainWindowLayout& layout,
    LayoutSplitter splitter, float delta)
{
    if (!std::isfinite(delta) || delta == 0) return false;
    float* preference;
    float size, minimum, maximum;
    if (splitter == LayoutSplitter::Toolbox)
    {
        if (layout.toolbox.width <= 0) return false;
        preference = &preferences.toolboxWidth;
        size = layout.toolbox.width + delta;
        minimum = layout.minToolboxWidth; maximum = layout.maxToolboxWidth;
    }
    else if (splitter == LayoutSplitter::Inspector)
    {
        if (layout.frameControls.width <= 0) return false;
        preference = &preferences.inspectorWidth;
        size = layout.frameControls.width - delta;
        minimum = layout.minInspectorWidth; maximum = layout.maxInspectorWidth;
    }
    else
    {
        if (layout.pipelineCollapsed || layout.pipeline.height <= 0) return false;
        preference = &preferences.pipelineHeight;
        size = layout.pipeline.height - delta;
        minimum = layout.minPipelineHeight; maximum = layout.maxPipelineHeight;
    }
    const float next = std::clamp(size, minimum, maximum) / layout.scale;
    if (*preference == next) return false;
    *preference = next;
    return true;
}

inline SplitterDrag DrawLayoutSplitter(const char* id, const RectF& bounds, bool vertical)
{
    if (bounds.width <= 0 || bounds.height <= 0) return {};
    ImGui::SetCursorPos({bounds.x, bounds.y});
    ImGui::InvisibleButton(id, {bounds.width, bounds.height});
    const bool active = ImGui::IsItemActive(), hovered = ImGui::IsItemHovered();
    if (active || hovered)
        ImGui::SetMouseCursor(vertical ? ImGuiMouseCursor_ResizeEW : ImGuiMouseCursor_ResizeNS);
    ImGui::GetWindowDrawList()->AddRectFilled(ImGui::GetItemRectMin(), ImGui::GetItemRectMax(),
        ImGui::GetColorU32(active ? ImGuiCol_SeparatorActive : hovered ? ImGuiCol_SeparatorHovered : ImGuiCol_Separator));
    SplitterDrag result;
    if (active && ImGui::IsMouseDragging(ImGuiMouseButton_Left, 0))
        result.delta = vertical ? ImGui::GetIO().MouseDelta.x : ImGui::GetIO().MouseDelta.y;
    result.released = ImGui::IsItemDeactivated();
    return result;
}

inline bool BeginLayoutDrawer(const char* id, const char* title, const RectF& bounds, ImVec2 origin, bool& requested)
{
    if (requested) { ImGui::OpenPopup(id); requested = false; }
    if (!ImGui::IsPopupOpen(id)) return false;
    if (bounds.width <= 0 || bounds.height <= 0)
    {
        if (ImGui::BeginPopup(id)) { ImGui::CloseCurrentPopup(); ImGui::EndPopup(); }
        return false;
    }
    ImGui::SetNextWindowPos({origin.x + bounds.x, origin.y + bounds.y}, ImGuiCond_Always);
    ImGui::SetNextWindowSize({bounds.width, bounds.height}, ImGuiCond_Always);
    if (!ImGui::BeginPopup(id, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize)) return false;
    ImGui::TextUnformatted(title);
    ImGui::SameLine();
    if (ImGui::SmallButton("Close")) ImGui::CloseCurrentPopup();
    ImGui::Separator();
    return true; // Caller pairs a true result with EndPopup().
}

ACMB_GUI_NAMESPACE_END
