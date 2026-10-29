#pragma once

#include "PipelineEditing.h"
#include "MainWindowLayout.h"
#include "ScrollablePanels.h"
#include "imgui.h"
#include <atomic>

ACMB_GUI_NAMESPACE_BEGIN

// Shared ImGui interaction paths. Tooltips and preview invalidation remain
// with the caller; activation and lock checks are identical in tests and GUI.
template<class Action>
bool CommandButton(const char* label, ImVec2 size, bool locked, Action action)
{
    if (ImGui::Button(label, size) && !locked)
    {
        action();
        return true;
    }
    return false;
}

inline bool RadioSetting(const char* label, int& value, int option, bool locked)
{
    if (ImGui::RadioButton(label, value == option) && !locked)
    {
        value = option;
        return true;
    }
    return false;
}

inline void ProcessSelectionKeys(PipelineSelection& selection, int count, int visibleCells)
{
    if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow))
        selection.MoveLeft(count, visibleCells);
    if (ImGui::IsKeyPressed(ImGuiKey_RightArrow))
        selection.MoveRight(count, visibleCells);
}

inline void ProcessSelectionMouse(PipelineSelection& selection, int count,
    const MainWindowLayout& layout, ImVec2 windowOrigin,
    float horizontalScroll = 0, int firstColumn = -1)
{
    if (!ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) || layout.cellWidth <= 0)
        return;
    const auto mouse = ImGui::GetMousePos();
    const float x = mouse.x - windowOrigin.x;
    const float y = mouse.y - windowOrigin.y;
    if (y < layout.pipeline.y || y > layout.pipeline.GetBottom() ||
        x < layout.pipeline.x || x >= layout.pipeline.GetRight())
        return;
    const float gridX = x - layout.pipeline.x + horizontalScroll;
    const int column = int(gridX / layout.cellWidth);
    const float inset = (layout.cellWidth - layout.cardSize.x) * 0.5f;
    const bool selected = gridX - column * layout.cellWidth > inset;
    selection.Select(column + (firstColumn < 0 ? selection.firstVisible : firstColumn), selected, count, layout.visibleCells);
}

// Stop requests cancellation; the session remains busy until the worker has
// actually completed. Only the cancellation flag is accessed by worker threads.
class ProcessingState
{
    bool _busy = false;
    std::atomic<bool> _stopRequested{false};

public:
    bool Begin()
    {
        if (_busy)
            return false;
        _stopRequested.store(false);
        _busy = true;
        return true;
    }
    void RequestStop() { if (_busy) _stopRequested.store(true); }
    void RequestShutdown() { _stopRequested.store(true); }
    void Finish() { _busy = false; }
    bool IsBusy() const { return _busy; }
    bool IsCancellationRequested() const { return _stopRequested.load(); }
};

ACMB_GUI_NAMESPACE_END
