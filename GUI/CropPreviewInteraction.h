#pragma once
#include "imgui.h"
#include <algorithm>
#include <cmath>

namespace acmb::gui
{
// Rectangles use image coordinates; handle hit areas use screen coordinates.
template<class Rectangle, class ImageSize>
bool DrawCropBorderHandles(Rectangle& rect, Rectangle& dragStartRect, const ImageSize& imageSize,
    ImVec2 topLeftPos, ImVec2 previewSize)
{
    const auto before = rect;

    const ImVec2 scale{previewSize.x / imageSize.width, previewSize.y / imageSize.height};
    const float left = topLeftPos.x + rect.x * scale.x;
    const float top = topLeftPos.y + rect.y * scale.y;
    const float right = left + rect.width * scale.x;
    const float bottom = top + rect.height * scale.y;
    const float radius = 5.0f * ImGui::GetFontSize() / 13.0f;
    const float cornerRadius = 12.0f * ImGui::GetFontSize() / 13.0f;
    // Corners own their entire hit area. Inset border handles so a border
    // hovered in the previous frame cannot steal a corner click.
    const ImVec2 positions[] = {
        {left - cornerRadius, top - cornerRadius}, {right - cornerRadius, top - cornerRadius},
        {left - cornerRadius, bottom - cornerRadius}, {right - cornerRadius, bottom - cornerRadius},
        {left - radius, top + cornerRadius}, {right - radius, top + cornerRadius},
        {left + cornerRadius, top - radius}, {left + cornerRadius, bottom - radius}};
    const ImVec2 sizes[] = {
        {2 * cornerRadius, 2 * cornerRadius}, {2 * cornerRadius, 2 * cornerRadius},
        {2 * cornerRadius, 2 * cornerRadius}, {2 * cornerRadius, 2 * cornerRadius},
        {2 * radius, bottom - top - 2 * cornerRadius}, {2 * radius, bottom - top - 2 * cornerRadius},
        {right - left - 2 * cornerRadius, 2 * radius}, {right - left - 2 * cornerRadius, 2 * radius}};
    const char* ids[] = {"##CropTopLeft", "##CropTopRight", "##CropBottomLeft", "##CropBottomRight",
        "##CropLeft", "##CropRight", "##CropTop", "##CropBottom"};
    for (int handle = 0; handle < 8; ++handle)
    {
        if (sizes[handle].x <= 0 || sizes[handle].y <= 0)
            continue;
        ImGui::SetCursorScreenPos(positions[handle]);
        ImGui::InvisibleButton(ids[handle], sizes[handle]);
        if (ImGui::IsItemHovered() || ImGui::IsItemActive())
            ImGui::SetMouseCursor(handle < 4
                ? (handle == 0 || handle == 3 ? ImGuiMouseCursor_ResizeNWSE : ImGuiMouseCursor_ResizeNESW)
                : (handle < 6 ? ImGuiMouseCursor_ResizeEW : ImGuiMouseCursor_ResizeNS));
        if (ImGui::IsItemActivated())
            dragStartRect = rect;
        if (!ImGui::IsItemActive())
            continue;

        const auto& io = ImGui::GetIO();
        const int dx = int(std::round((io.MousePos.x - io.MouseClickedPos[0].x) / scale.x));
        const int dy = int(std::round((io.MousePos.y - io.MouseClickedPos[0].y) / scale.y));
        const int startRight = dragStartRect.x + dragStartRect.width;
        const int startBottom = dragStartRect.y + dragStartRect.height;
        if (handle == 0 || handle == 2 || handle == 4)
        {
            rect.x = std::clamp(dragStartRect.x + dx, 0, startRight - 1);
            rect.width = startRight - rect.x;
        }
        else if (handle == 1 || handle == 3 || handle == 5)
            rect.width = std::clamp(startRight + dx, rect.x + 1, imageSize.width) - rect.x;
        if (handle == 0 || handle == 1 || handle == 6)
        {
            rect.y = std::clamp(dragStartRect.y + dy, 0, startBottom - 1);
            rect.height = startBottom - rect.y;
        }
        else if (handle == 2 || handle == 3 || handle == 7)
            rect.height = std::clamp(startBottom + dy, rect.y + 1, imageSize.height) - rect.y;
    }
    return before.x != rect.x || before.y != rect.y ||
        before.width != rect.width || before.height != rect.height;
}
}

