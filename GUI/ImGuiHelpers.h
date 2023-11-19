#pragma once
#include "imgui.h"
#include <string>

namespace ImGui
{
    void SetTooltipIfHovered( const std::string& text, float scaling );
}