#pragma once
#include "MenuItem.h"
#include "FontRegistry.h"
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <memory>

ACMB_GUI_NAMESPACE_BEGIN

using Items = std::unordered_map<std::string, std::map<int, std::unique_ptr<MenuItem>>>;

class MenuItemsHolder
{    
    Items _items;

private:
    MenuItemsHolder() = default;
    MenuItemsHolder( const MenuItemsHolder& ) = delete;
    MenuItemsHolder( MenuItemsHolder&& ) = delete;

public:

    static MenuItemsHolder& GetInstance()
    {
        static MenuItemsHolder instance;
        return instance;
    }

    const Items& GetItems()
    {
        return _items;
    }    

    bool AddItem( const std::string& category, uint8_t order, const std::string& icon, const std::string& tooltip, const std::function<void(Point)>& action)
    {
        auto it = _items.find( category );
        if ( it == _items.end() )
            it = _items.insert_or_assign( category, std::map<int, std::unique_ptr<MenuItem>>{} ).first;

        it->second.insert_or_assign(order, std::make_unique<MenuItem>( icon, tooltip, action ) );
        return true;
    }

};

#define REGISTER_MENU_ITEM( category, order, icon, tooltip, action ) \
static inline bool handle = MenuItemsHolder::GetInstance().AddItem( category, order, icon, tooltip, action ); \

#define REGISTER_TOOLS_ITEM( ToolClassType ) \
static inline bool handle = MenuItemsHolder::GetInstance().AddItem( "Tools", ToolClassType::order, ToolClassType::icon,  ToolClassType::tooltip, [](Point p){ MainWindow::GetInstance( FontRegistry::Instance() ).AddElementToGrid<ToolClassType>( p ); } );


ACMB_GUI_NAMESPACE_END