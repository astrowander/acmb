#pragma once

#include "MenuItemsHolder.h"
#include "ScrollablePanels.h"

ACMB_GUI_NAMESPACE_BEGIN

inline void DrawToolLibrary(const std::vector<ToolGroupView>& groups,
    const std::function<bool()>& isLocked, size_t position, bool replace,
    const std::function<void(const MenuItem&)>& tooltip = {},
    const std::function<void(const char*, int)>& observeItem = {}, bool closeOnAction = false)
{
    for (const auto& group : groups)
    {
        ImGui::PushID(int(group.group));
        const bool expanded = ImGui::CollapsingHeader(group.caption, ImGuiTreeNodeFlags_DefaultOpen);
        if (observeItem)
            observeItem(group.caption, -1);
        if (expanded)
        {
            for (const auto& [id, item] : group.tools)
            {
                ImGui::PushID(id);
                const bool locked = !item->unlockable &&
                    (isLocked() || ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId));
                const bool activated = IconTextButton("command", item->icon, item->caption,
                    ImGui::GetContentRegionAvail().x, locked,
                    [&] { item->action(position, replace); });
                if (activated && closeOnAction) ImGui::CloseCurrentPopup();
                if (observeItem)
                    observeItem(group.caption, id);
                if (tooltip)
                    tooltip(*item);
                ImGui::PopID();
            }
        }
        ImGui::PopID();
        ImGui::Spacing();
    }
}

ACMB_GUI_NAMESPACE_END
