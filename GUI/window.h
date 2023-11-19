#pragma once
#include "./../Core/macros.h"

#include "imgui.h"
#include <vector>
#include <memory>
#include <string>

ACMB_GUI_NAMESPACE_BEGIN

class Window : public std::enable_shared_from_this<Window>
{
private:
    static float GetMenuScaling();

protected:

    std::string _name;

    ImVec2 _pos;
    ImVec2 _size;

    bool _isOpen = false;
    virtual ImGuiWindowFlags flags() const { return ImGuiWindowFlags( 0 ); }

public:
    inline static const float cMenuScaling = GetMenuScaling();

    Window( const std::string& name, const ImVec2& size );
    virtual ~Window() = default;

    virtual void Show();
    virtual void DrawDialog() = 0;

    void SetPos( const ImVec2& pos );
};

ACMB_GUI_NAMESPACE_END
