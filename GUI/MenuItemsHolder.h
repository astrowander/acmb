#pragma once
#include "MenuItem.h"
#include "FontRegistry.h"
#include <map>
#include <memory>
#include <algorithm>
#include <array>
#include <vector>

ACMB_GUI_NAMESPACE_BEGIN

using Items = std::map<std::string, std::map<int, std::unique_ptr<MenuItem>>>;

struct ToolGroupView
{
    ToolGroup group;
    const char* caption;
    std::vector<std::pair<int, const MenuItem*>> tools;
};

class MenuItemsHolder
{    
    Items _items;

private:
    MenuItemsHolder() = default;
    MenuItemsHolder( const MenuItemsHolder& ) = delete;
    MenuItemsHolder( MenuItemsHolder&& ) = delete;

public:

    static MenuPresentation GetToolPresentation(uint8_t id)
    {
        // These are the existing .acmb IDs. Display order is independent of ID.
        switch (id)
        {
        case 1: return {ToolGroup::InputOutput, 0};
        case 2: return {ToolGroup::InputOutput, 1};
        case 8: return {ToolGroup::InputOutput, 2};
        case 3: return {ToolGroup::StackCalibration, 0};
        case 4: return {ToolGroup::StackCalibration, 1};
        case 5: return {ToolGroup::StackCalibration, 2};
        case 6: return {ToolGroup::Geometry, 0};
        case 7: return {ToolGroup::Geometry, 1};
        case 11: return {ToolGroup::Geometry, 2};
        case 14: return {ToolGroup::Geometry, 3};
        case 9: return {ToolGroup::Adjustments, 0};
        case 10: return {ToolGroup::Adjustments, 1};
        case 12: return {ToolGroup::Cleanup, 0};
        case 15: return {ToolGroup::Cleanup, 1};
        case 16: return {ToolGroup::Cleanup, 2};
        default: return {ToolGroup::Other, id};
        }
    }

    static std::vector<ToolGroupView> BuildToolGroups(const Items& items)
    {
        std::vector<ToolGroupView> groups{
            {ToolGroup::InputOutput, "Input / Output", {}},
            {ToolGroup::StackCalibration, "Stack & Calibration", {}},
            {ToolGroup::Geometry, "Geometry", {}},
            {ToolGroup::Adjustments, "Adjustments", {}},
            {ToolGroup::Cleanup, "Cleanup", {}},
            {ToolGroup::Other, "Other", {}}};
        const auto tools = items.find("Tools");
        if (tools == items.end())
            return {};
        for (const auto& [id, item] : tools->second)
            for (auto& group : groups)
                if (group.group == item->presentation.group)
                    group.tools.emplace_back(id, item.get());
        for (auto& group : groups)
            std::sort(group.tools.begin(), group.tools.end(), [](const auto& a, const auto& b) {
                if (a.second->presentation.order != b.second->presentation.order)
                    return a.second->presentation.order < b.second->presentation.order;
                return a.first < b.first;
            });
        std::erase_if(groups, [](const auto& group) { return group.tools.empty(); });
        return groups;
    }

    std::vector<ToolGroupView> GetToolGroups() const { return BuildToolGroups(_items); }

    static MenuItemsHolder& GetInstance()
    {
        static MenuItemsHolder instance;
        return instance;
    }

    const Items& GetItems()
    {
        return _items;
    }    

    bool AddItem( const std::string& category, uint8_t order, const std::string& icon, const std::string& caption, const std::string& tooltip, const std::function<void(size_t, bool)>& action, bool unlockable = false )
    {
        auto it = _items.find( category );
        if ( it == _items.end() )
            it = _items.insert_or_assign( category, std::map<int, std::unique_ptr<MenuItem>>{} ).first;

        const auto presentation = category == "Tools" ? GetToolPresentation(order) : MenuPresentation{};
        it->second.insert_or_assign(order, std::make_unique<MenuItem>( icon, caption, tooltip, action, unlockable, presentation ) );
        return true;
    }

};

#define REGISTER_MENU_ITEM( category, order, icon, caption, tooltip, action ) \
static inline bool handle = MenuItemsHolder::GetInstance().AddItem( category, order, icon, caption, tooltip, action ); \

#define REGISTER_TOOLS_ITEM( ToolClassType ) \
static inline bool handle = MenuItemsHolder::GetInstance().AddItem( "Tools", ToolClassType::order, ToolClassType::icon,  ToolClassType::caption, ToolClassType::tooltip, [](size_t pos, bool replace){ MainWindow::GetInstance( FontRegistry::Instance() ).AddElementToGrid<ToolClassType>( pos, replace ); } );


ACMB_GUI_NAMESPACE_END
