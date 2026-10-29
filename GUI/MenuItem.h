#pragma once
#include "./../Core/macros.h"
#include "./../Geometry/point.h"
#include <string>
#include <functional>

ACMB_GUI_NAMESPACE_BEGIN

enum class ToolGroup { InputOutput, StackCalibration, Geometry, Adjustments, Cleanup, Other };

// Presentation only: never used as a serialized tool identifier.
struct MenuPresentation
{
    ToolGroup group = ToolGroup::Other;
    int order = 0;
};

struct MenuItem
{
    std::string icon;
    std::string caption;
    std::string tooltip;
    std::function<void(size_t, bool)> action;
    bool unlockable = false;
    MenuPresentation presentation;


    MenuItem( const std::string& icon, const std::string& caption, const std::string& tooltip, const std::function<void(size_t, bool)> action, bool unlockable = false, MenuPresentation presentation = {} )
    : icon( icon )
    , caption( caption )
    , tooltip( tooltip )
    , action( action )
    , unlockable( unlockable )
    , presentation( presentation )
    { }
};

ACMB_GUI_NAMESPACE_END
