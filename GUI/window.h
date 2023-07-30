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

    std::vector<std::shared_ptr<Window>> _children;
    std::weak_ptr<Window> _pParent;

    bool _isOpen = false;

    float menuScaling = 1.0f;

public:
    inline static const float cMenuScaling = GetMenuScaling();

    Window( const std::string& name, const ImVec2& pos, const ImVec2& size, std::shared_ptr<Window> pParent = nullptr );
    virtual ~Window() = default;

    void Show( ImGuiWindowFlags flags = 0 );
    virtual void DrawDialog() = 0;
};

ACMB_GUI_NAMESPACE_END
