#pragma once

#include "../Core/macros.h"
#include "imgui.h"
#include <algorithm>
#include <string>

ACMB_GUI_NAMESPACE_BEGIN

inline void CenteredCardCaption(const char* windowName)
{
    const std::string name(windowName);
    const auto caption = name.substr(0, name.find("##"));
    const auto windowPosition = ImGui::GetWindowPos();
    const ImVec2 position{windowPosition.x + 1, windowPosition.y + 1};
    const float width = ImGui::GetWindowWidth() - 2;
    const float height = 24;
    const auto textSize = ImGui::CalcTextSize(caption.c_str());
    auto* drawList = ImGui::GetWindowDrawList();
    drawList->PushClipRect(position, {position.x + width, position.y + height}, true);
    drawList->AddRectFilled(position, {position.x + width, position.y + height},
        ImGui::GetColorU32(ImGuiCol_TitleBgActive));
    drawList->AddText({position.x + std::max(0.0f, (width - textSize.x) * 0.5f),
        position.y + (height - textSize.y) * 0.5f}, ImGui::GetColorU32(ImGuiCol_Text), caption.c_str());
    drawList->PopClipRect();
}

inline bool BeginVerticalPanel(const char* id, ImVec2 size, float minimumContentWidth, bool borders = true)
{
    // Reserve a scrollbar even before content overflows, so controls do not
    // change width when a tool adds settings or an error message appears.
    ImGui::SetNextWindowContentSize({minimumContentWidth, 0});
    return ImGui::BeginChild(id, size, borders ? ImGuiChildFlags_Borders : ImGuiChildFlags_None,
        ImGuiWindowFlags_AlwaysVerticalScrollbar | ImGuiWindowFlags_HorizontalScrollbar);
}

inline bool BeginPipelinePanel(const char* id, ImVec2 size, int columns, float cellWidth)
{
    ImGui::SetNextWindowContentSize({columns * cellWidth, 1.0f});
    return ImGui::BeginChild(id, size, ImGuiChildFlags_None,
        ImGuiWindowFlags_HorizontalScrollbar | ImGuiWindowFlags_AlwaysHorizontalScrollbar);
}

inline ImVec2 MinimumButtonSize(const char* label, ImVec2 requested)
{
    const auto& style = ImGui::GetStyle();
    const auto text = ImGui::CalcTextSize(label, nullptr, true);
    const float minimumHeight = std::max(32.0f, ImGui::GetFrameHeight());
    // Negative widths mean "fill available space" in ImGui. Resolve them
    // before applying the minimum, preserving that behavior for existing tools.
    const float width = requested.x < 0 ? ImGui::GetContentRegionAvail().x : requested.x;
    return {std::max(width, text.x + 2 * style.FramePadding.x),
        std::max(requested.y, minimumHeight)};
}

// The icon is merged into the text font. ImGui owns layout and hit testing for
// the whole icon-and-caption row, including long captions and keyboard focus.
template<class Action>
bool IconTextButton(const char* id, const std::string& icon, const std::string& caption,
    float width, bool locked, Action action)
{
    const std::string label = icon + "  " + caption + "###" + id;
    if (ImGui::Button(label.c_str(), MinimumButtonSize(label.c_str(), {width, 36})) && !locked)
    {
        action();
        return true;
    }
    return false;
}

ACMB_GUI_NAMESPACE_END
