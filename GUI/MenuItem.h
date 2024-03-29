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


    MenuItem( const std::string& icon, const std::string& caption, const std::string& tooltip, const std::function<void(Point)> action )
    : icon( icon )
    , caption( caption )
    , tooltip( tooltip )
    , action( action )
    { }
};

ACMB_GUI_NAMESPACE_END
