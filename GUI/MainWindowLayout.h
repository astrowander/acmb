#pragma once

#include "../Core/macros.h"
#include "../Geometry/rect.h"
#include "imgui.h"
#include <algorithm>

ACMB_GUI_NAMESPACE_BEGIN

// Pure panel geometry: all coordinates are local to MainWindow. No monitor,
// singleton, renderer, or ImGui context is needed to calculate these bounds.
struct MainWindowLayout
{
    inline static constexpr float MenuWidth = 320.0f;
    inline static constexpr float InspectorWidth = 340.0f;
    inline static constexpr float HeaderHeight = 25.0f;
    inline static constexpr float GridLeft = 25.0f;
    inline static constexpr float CellMinWidth = 200.0f;
    inline static constexpr float CellHeight = 300.0f;

    RectF toolbox;
    RectF preview;
    RectF frameControls;
    RectF pipelineHeader;
    RectF pipeline;
    float toolboxSeparatorX;
    float inspectorSeparatorX;
    int visibleCells;
    float cellWidth;

    static int CountVisibleCells(float windowWidth)
    {
        return std::max(1, int(std::max(0.0f, windowWidth - GridLeft) / CellMinWidth));
    }

    static MainWindowLayout Calculate(ImVec2 size, ImVec2 spacing,
        ImVec2 windowPadding, float textLineHeightWithSpacing)
    {
        const float width = std::max(0.0f, size.x);
        const float height = std::max(0.0f, size.y);
        const float gridBottom = std::clamp(height - 3.0f - textLineHeightWithSpacing, 0.0f, height);
        const float gridTop = std::max(0.0f, gridBottom - CellHeight);
        const float headerTop = std::max(0.0f, gridTop - HeaderHeight);

        MainWindowLayout result{};
        result.pipeline = {std::min(GridLeft, width), gridTop,
            std::max(0.0f, width - GridLeft), gridBottom - gridTop};
        result.pipelineHeader = {0, headerTop, width, gridTop - headerTop};
        result.visibleCells = CountVisibleCells(width);
        result.cellWidth = result.pipeline.width / result.visibleCells;

        const float toolboxX = std::min(windowPadding.x, width);
        const float toolboxY = std::min(windowPadding.y, headerTop);
        result.toolbox = {toolboxX, toolboxY,
            std::min(MenuWidth + 2.0f * spacing.x, width - toolboxX),
            std::max(0.0f, headerTop - spacing.y - toolboxY)};
        result.toolboxSeparatorX = std::min(MenuWidth + 4.0f * spacing.x, width);
        const float previewLeft = std::min(MenuWidth + 6.0f * spacing.x, width);
        const float previewRight = std::clamp(width - InspectorWidth - 4.0f * spacing.x, previewLeft, width);
        const float previewTop = std::min(2.0f * spacing.y, headerTop);
        const float previewBottom = std::clamp(headerTop - 2.0f * spacing.y, previewTop, headerTop);
        result.preview = {previewLeft, previewTop, previewRight - previewLeft, previewBottom - previewTop};

        const float inspectorX = std::clamp(width - InspectorWidth + 2.0f * spacing.x, previewRight, width);
        result.frameControls = {inspectorX, std::min(spacing.y, headerTop),
            std::clamp(InspectorWidth - 4.0f * spacing.x, 0.0f, width - inspectorX),
            result.preview.height};
        result.inspectorSeparatorX = std::clamp(width - InspectorWidth - 2.0f * spacing.x,
            result.toolboxSeparatorX, width);
        return result;
    }
};

ACMB_GUI_NAMESPACE_END
