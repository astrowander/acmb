#pragma once

#include "GuiInteractions.h"
#include "MenuItem.h"
#include <array>
#include <functional>

ACMB_GUI_NAMESPACE_BEGIN

struct ToolbarActions
{
    const MenuItem* open;
    const MenuItem* save;
    const MenuItem* run;
    const MenuItem* stop;
    const MenuItem* help;
};

inline void DrawTopToolbar(ImVec2 size, const ToolbarActions& actions,
    const std::function<bool()>& isLocked, size_t position, bool replace,
    bool cudaAvailable, bool& enableCuda,
    const std::function<void(const char*)>& observeItem = {},
    const std::function<void()>& drawLayoutMenu = {})
{
    const auto observe = [&](const char* id) { if (observeItem) observeItem(id); };
    if (ImGui::BeginChild("##TopToolbar", size, ImGuiChildFlags_None,
        ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoScrollbar))
    {
        if (ImGui::BeginMenuBar())
        {
            const std::array<const MenuItem*, 4> commands{actions.open, actions.save, actions.run, actions.stop};
            for (const auto* item : commands)
            {
                const bool blocked = !item->unlockable &&
                    (isLocked() || ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId));
                const auto label = item->icon + "  " + item->caption + "###Toolbar" + item->caption;
                ImGui::BeginDisabled(blocked);
                CommandButton(label.c_str(), {0, 0}, blocked, [&] { item->action(position, replace); });
                observe(item->caption.c_str());
                ImGui::EndDisabled();
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                    ImGui::SetTooltip("%s", item->tooltip.c_str());
                ImGui::SameLine();
            }
            ImGui::Separator();
            const bool settingsOpen = ImGui::BeginMenu("Settings", !isLocked());
            observe("Settings");
            if (settingsOpen)
            {
                ImGui::MenuItem("Enable CUDA", nullptr, &enableCuda, cudaAvailable && !isLocked());
                observe("Enable CUDA");
                ImGui::EndMenu();
            }
            const bool helpOpen = ImGui::BeginMenu("Help", !isLocked());
            observe("Help");
            if (helpOpen)
            {
                if (ImGui::MenuItem("Instructions"))
                    actions.help->action(position, replace);
                observe("Instructions");
                ImGui::EndMenu();
            }
            if (drawLayoutMenu) drawLayoutMenu();
            ImGui::EndMenuBar();
        }
    }
    ImGui::EndChild();
}

ACMB_GUI_NAMESPACE_END
