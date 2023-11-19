#pragma once
#include "imgui.h"
#include <string>
#include <vector>

namespace ImGui
{
    void SetTooltipIfHovered( const std::string& text, float scaling );

    enum class ModalMessageType
    {
        Error,
        Success,
        Help
    };

    void ShowModalMessage( const std::vector<std::string>& msg, ModalMessageType msgType, bool& isEnabled );
    
}