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
    inline static constexpr float HeaderHeight = 40.0f;
    inline static constexpr float CommandsWidth = 160.0f;
    inline static constexpr float GridLeft = CommandsWidth + 16.0f;
    inline static constexpr float MinimumWorkspaceWidth = 1000.0f;
    inline static constexpr float MinimumWorkspaceHeight = 600.0f;
    inline static constexpr float CardWidth = 280.0f;
    inline static constexpr float CardHeight = 250.0f;
    inline static constexpr float CellPadding = 25.0f;
    inline static constexpr float CellMinWidth = CardWidth + 2 * CellPadding;
    inline static constexpr float CellHeight = 300.0f;

    RectF toolbox;
    ImVec2 canvasSize;
    ImVec2 cardSize;
    RectF preview;
    RectF frameControls;
    RectF pipelineHeader;
    RectF pipelineCommands;
    RectF pipeline;
    float toolboxSeparatorX;
    float inspectorSeparatorX;
    int visibleCells;
    float cellWidth;

    static int CountVisibleCells(float windowWidth, float minimumCellWidth = CellMinWidth)
    {
        return std::max(1, int(std::max(0.0f, windowWidth - GridLeft) / minimumCellWidth));
    }

    static MainWindowLayout Calculate(ImVec2 size, ImVec2 spacing,
        ImVec2 windowPadding, float textLineHeightWithSpacing)
    {
        MainWindowLayout result{};
        result.visibleCells = 1;
        if (size.x <= 0 || size.y <= 0)
            return result;
        const float scale = std::max(1.0f, (textLineHeightWithSpacing - spacing.y) / 15.0f);
        const float width = std::max(MinimumWorkspaceWidth * scale, size.x);
        const float height = std::max(MinimumWorkspaceHeight * scale, size.y);
        result.canvasSize = {width, height};
        result.cardSize = {CardWidth * scale, CardHeight * scale};
        const float menuWidth = MenuWidth * scale;
        const float inspectorWidth = InspectorWidth * scale;
        const float gridBottom = std::clamp(height - 3.0f - textLineHeightWithSpacing, 0.0f, height);
        const float gridTop = std::max(0.0f, gridBottom - CellHeight * scale);
        const float headerTop = std::max(0.0f, gridTop - HeaderHeight);

        result.pipelineCommands = {windowPadding.x, gridTop, CommandsWidth * scale, gridBottom - gridTop};
        const float pipelineLeft = result.pipelineCommands.GetRight() + spacing.x;
        result.pipeline = {pipelineLeft, gridTop,
            std::max(0.0f, width - pipelineLeft - windowPadding.x), gridBottom - gridTop};
        result.pipelineHeader = {0, headerTop, width, gridTop - headerTop};
        result.cellWidth = CellMinWidth * scale;
        result.visibleCells = std::max(1, int((result.pipeline.width - 2 * windowPadding.x) / result.cellWidth));

        const float toolboxX = std::min(windowPadding.x, width);
        const float toolboxY = std::min(windowPadding.y, headerTop);
        result.toolbox = {toolboxX, toolboxY,
            std::min(menuWidth + 2.0f * spacing.x, width - toolboxX),
            std::max(0.0f, headerTop - spacing.y - toolboxY)};
        result.toolboxSeparatorX = std::min(menuWidth + 4.0f * spacing.x, width);
        const float previewLeft = std::min(menuWidth + 6.0f * spacing.x, width);
        const float previewRight = std::clamp(width - inspectorWidth - 4.0f * spacing.x, previewLeft, width);
        const float previewTop = std::min(2.0f * spacing.y, headerTop);
        const float previewBottom = std::clamp(headerTop - 2.0f * spacing.y, previewTop, headerTop);
        result.preview = {previewLeft, previewTop, previewRight - previewLeft, previewBottom - previewTop};

        const float inspectorX = std::clamp(width - inspectorWidth + 2.0f * spacing.x, previewRight, width);
        result.frameControls = {inspectorX, std::min(spacing.y, headerTop),
            std::clamp(inspectorWidth - 4.0f * spacing.x, 0.0f, width - inspectorX),
            result.preview.height};
        result.inspectorSeparatorX = std::clamp(width - inspectorWidth - 2.0f * spacing.x,
            result.toolboxSeparatorX, width);
        return result;
    }
};

ACMB_GUI_NAMESPACE_END
