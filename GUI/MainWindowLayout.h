#pragma once

#include "../Core/macros.h"
#include "../Geometry/rect.h"
#include "LayoutPreferences.h"
#include "imgui.h"
#include <algorithm>

ACMB_GUI_NAMESPACE_BEGIN

enum class PanelMode { Wide, Medium, Narrow };

struct MainWindowLayout
{
    inline static constexpr float MinimumToolboxWidth = LayoutPreferences::MinimumToolboxWidth;
    inline static constexpr float MinimumInspectorWidth = LayoutPreferences::MinimumInspectorWidth;
    inline static constexpr float MinimumPreviewWidth = 320;
    inline static constexpr float MinimumPreviewHeight = 120;
    inline static constexpr float SplitterWidth = 6;
    inline static constexpr float HeaderHeight = 32;
    inline static constexpr float CommandsWidth = 160;
    inline static constexpr float GridLeft = CommandsWidth + 16;
    // Hard fallback only for windows too small to fit even the toolbar.
    inline static constexpr float MinimumWorkspaceWidth = 600;
    inline static constexpr float MinimumWorkspaceHeight = 480;
    inline static constexpr float CardWidth = 200;
    inline static constexpr float CardHeight = 110;
    inline static constexpr float CellPadding = 25;
    inline static constexpr float CellMinWidth = CardWidth + 2 * CellPadding;
    inline static constexpr float CellHeight = CardHeight + 2 * CellPadding;

    PanelMode mode = PanelMode::Wide;
    RectF toolbox, toolbar, drawerBar, preview, frameControls;
    RectF toolsDrawer, inspectorDrawer;
    RectF toolboxSplitter, inspectorSplitter, pipelineSplitter;
    RectF pipelineHeader, pipelineCommands, pipeline, pipelineProgress;
    ImVec2 canvasSize{}, cardSize{};
    float toolboxSeparatorX = 0, inspectorSeparatorX = 0;
    int visibleCells = 1;
    float cellWidth = 0, scale = 1;
    float minToolboxWidth = 0, maxToolboxWidth = 0;
    float minInspectorWidth = 0, maxInspectorWidth = 0;
    float minPipelineHeight = 0, maxPipelineHeight = 0;
    bool pipelineCollapsed = false;

    RectF FitPreview(ImVec2 image) const
    {
        if (!std::isfinite(image.x) || !std::isfinite(image.y) || image.x <= 0 || image.y <= 0 ||
            preview.width <= 0 || preview.height <= 0) return {};
        const float factor = std::min({1.0f, preview.width / image.x, preview.height / image.y});
        const float width = image.x * factor, height = image.y * factor;
        return {preview.x + (preview.width - width) / 2, preview.y + (preview.height - height) / 2, width, height};
    }

    static int CountVisibleCells(float width, float cell = CellMinWidth)
    {
        return std::max(1, int(std::max(0.0f, width - GridLeft) / cell));
    }

    static MainWindowLayout Calculate(ImVec2 size, ImVec2 spacing, ImVec2 padding,
        float textLineHeight, const LayoutPreferences& preferences = {})
    {
        MainWindowLayout r;
        if (size.x <= 0 || size.y <= 0) return r;
        auto prefs = preferences;
        prefs.Clamp();
        r.scale = std::max(1.0f, (textLineHeight - spacing.y) / 15.0f);
        const float s = r.scale;
        const float width = std::max(MinimumWorkspaceWidth * s, size.x);
        const float height = std::max(MinimumWorkspaceHeight * s, size.y);
        const float bodyX = padding.x, bodyWidth = width - 2 * padding.x;
        const float split = SplitterWidth * s, gap = spacing.x;
        const float previewMin = MinimumPreviewWidth * s;
        r.canvasSize = {width, height};
        r.cardSize = {CardWidth * s, CardHeight * s};
        r.cellWidth = CellMinWidth * s;
        r.minToolboxWidth = MinimumToolboxWidth * s;
        r.minInspectorWidth = MinimumInspectorWidth * s;
        const float wideMinimum = r.minToolboxWidth + previewMin + r.minInspectorWidth + 2 * split + 4 * gap;
        const float mediumMinimum = previewMin + r.minInspectorWidth + split + 2 * gap;
        r.mode = bodyWidth >= wideMinimum ? PanelMode::Wide :
            bodyWidth >= mediumMinimum ? PanelMode::Medium : PanelMode::Narrow;

        r.toolbar = {padding.x, padding.y, bodyWidth,
            std::max(36.0f * s, textLineHeight + 2 * padding.y)};
        float top = r.toolbar.GetBottom() + spacing.y;
        if (r.mode != PanelMode::Wide)
        {
            r.drawerBar = {bodyX, top, bodyWidth, 30 * s};
            top = r.drawerBar.GetBottom() + spacing.y;
        }
        const float progressHeight = std::max(20.0f * s, textLineHeight);
        r.pipelineProgress = {bodyX, height - padding.y - progressHeight, bodyWidth, progressHeight};
        const float bottom = r.pipelineProgress.y - spacing.y;
        r.pipelineCollapsed = prefs.pipelineCollapsed;
        r.minPipelineHeight = CellHeight * s;
        r.maxPipelineHeight = std::max(r.minPipelineHeight,
            std::min(LayoutPreferences::MaximumPipelineHeight * s,
                bottom - top - MinimumPreviewHeight * s - HeaderHeight * s - split - 2 * spacing.y));
        const float pipelineHeight = prefs.pipelineCollapsed ? 64 * s :
            std::clamp(prefs.pipelineHeight * s, r.minPipelineHeight, r.maxPipelineHeight);
        const float pipelineTop = bottom - pipelineHeight;
        const float headerTop = pipelineTop - HeaderHeight * s;
        const float bodyBottom = headerTop - split - 2 * spacing.y;
        const float bodyHeight = std::max(0.0f, bodyBottom - top);
        r.pipelineHeader = {bodyX, headerTop, bodyWidth, HeaderHeight * s};
        r.pipelineSplitter = {bodyX, bodyBottom + spacing.y, bodyWidth, split};
        r.pipelineCommands = {bodyX, pipelineTop, CommandsWidth * s, pipelineHeight};
        const float pipelineLeft = r.pipelineCommands.GetRight() + gap;
        r.pipeline = {pipelineLeft, pipelineTop, width - padding.x - pipelineLeft, pipelineHeight};
        r.visibleCells = std::max(1, int((r.pipeline.width - 2 * padding.x) / r.cellWidth));

        float left = bodyX, right = width - padding.x;
        if (r.mode != PanelMode::Narrow)
        {
            const float leftReserve = r.mode == PanelMode::Wide ? r.minToolboxWidth + split + 2 * gap : 0;
            r.maxInspectorWidth = std::min(LayoutPreferences::MaximumPanelWidth * s,
                bodyWidth - previewMin - split - 2 * gap - leftReserve);
            const float inspectorWidth = std::clamp(prefs.inspectorWidth * s, r.minInspectorWidth, r.maxInspectorWidth);
            r.frameControls = {right - inspectorWidth, top, inspectorWidth, bodyHeight};
            r.inspectorSplitter = {r.frameControls.x - gap - split, top, split, bodyHeight};
            right = r.inspectorSplitter.x - gap;
        }
        if (r.mode == PanelMode::Wide)
        {
            r.maxToolboxWidth = std::min(LayoutPreferences::MaximumPanelWidth * s,
                right - bodyX - previewMin - split - 2 * gap);
            const float toolboxWidth = std::clamp(prefs.toolboxWidth * s, r.minToolboxWidth, r.maxToolboxWidth);
            r.toolbox = {bodyX, top, toolboxWidth, bodyHeight};
            r.toolboxSplitter = {r.toolbox.GetRight() + gap, top, split, bodyHeight};
            left = r.toolboxSplitter.GetRight() + gap;
            // The other splitter's upper bound must also respect the actual toolbox width.
            r.maxInspectorWidth = std::min(LayoutPreferences::MaximumPanelWidth * s,
                bodyWidth - toolboxWidth - previewMin - 2 * split - 4 * gap);
        }
        r.preview = {left, top, std::max(0.0f, right - left), bodyHeight};
        r.toolboxSeparatorX = r.toolboxSplitter.x;
        r.inspectorSeparatorX = r.inspectorSplitter.x;
        const float drawerSpace = std::max(0.0f, size.x - 2 * padding.x);
        const float drawerTop = std::min(top, std::max(0.0f, size.y - padding.y));
        const float drawerHeight = std::max(0.0f, std::min(bodyHeight, size.y - padding.y - drawerTop));
        if (r.mode != PanelMode::Wide)
            r.toolsDrawer = {bodyX, drawerTop,
                std::min(drawerSpace, std::max(r.minToolboxWidth, prefs.toolboxWidth * s)), drawerHeight};
        if (r.mode == PanelMode::Narrow)
        {
            const float drawerWidth = std::min(drawerSpace, std::max(r.minInspectorWidth, prefs.inspectorWidth * s));
            r.inspectorDrawer = {std::max(bodyX, size.x - padding.x - drawerWidth), drawerTop, drawerWidth, drawerHeight};
        }
        return r;
    }
};

ACMB_GUI_NAMESPACE_END
