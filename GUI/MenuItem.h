#pragma once
#include "./../Core/macros.h"
#include "./../Geometry/point.h"
#include <string>
#include <functional>

ACMB_GUI_NAMESPACE_BEGIN

struct MenuItem
{
    std::string icon;
    std::string caption;
    std::string tooltip;
    std::function<void(Point)> action;
    bool unlockable = false;


    MenuItem( const std::string& icon, const std::string& caption, const std::string& tooltip, const std::function<void(Point)> action, bool unlockable = false )
    : icon( icon )
    , caption( caption )
    , tooltip( tooltip )
    , action( action )
    , unlockable( unlockable )
    { }
};

ACMB_GUI_NAMESPACE_END
