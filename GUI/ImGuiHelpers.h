#pragma once
#include "imgui.h"

#include <string>
#include <vector>
#include <functional>
#include <memory>

#include "./../Core/macros.h"

ACMB_GUI_NAMESPACE_BEGIN
class PipelineElementWindow;
ACMB_GUI_NAMESPACE_END

namespace UI
{
    void SetTooltipIfHovered( const std::string& text, float scaling );

    enum class ModalMessageType
    {
        Error,
        Success,
        Help
    };

    void ShowModalMessage( const std::vector<std::string>& msg, ModalMessageType msgType, bool& isEnabled );

    bool Button( const std::string& name, const ImVec2& size, std::function<void()> action, const std::string & tooltip, acmb::gui::PipelineElementWindow* parent = nullptr );

    bool UnlockableButton( const std::string& name, const ImVec2& size, std::function<void()> action, const std::string& tooltip, acmb::gui::PipelineElementWindow* parent = nullptr );

    bool RadioButton( const std::string& label, int* v, int v_button, const std::string& tooltip, acmb::gui::PipelineElementWindow* parent = nullptr );

    bool Checkbox( const std::string& label, bool* v, const std::string& tooltip, acmb::gui::PipelineElementWindow* parent = nullptr );

    bool DragInt( const std::string& label, int* v, float v_speed, int v_min, int v_max,  const std::string& tooltip, acmb::gui::PipelineElementWindow* parent = nullptr );

    bool DragInt2( const std::string& label, int v[2], float v_speed, int v_min, int v_max, const std::string& tooltip, acmb::gui::PipelineElementWindow* parent = nullptr );
    bool DragFloat( const std::string& label, float* v, float v_speed, float v_min, float v_max, const std::string& tooltip, acmb::gui::PipelineElementWindow* parent = nullptr );

    bool Combo( const std::string& label, int* current_item, const std::string& items_separated_by_zeros, const std::string& tooltip, acmb::gui::PipelineElementWindow* parent = nullptr );

    bool InputInt( const std::string& label, int* v, int step, int step_fast, int min, int max, const std::string& tooltip, acmb::gui::PipelineElementWindow* parent = nullptr );
}