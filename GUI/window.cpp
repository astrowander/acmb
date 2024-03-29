#include "window.h"

#ifdef _WIN32
#include <Windows.h>
#elif defined ( __linux__ )
#include <GLFW/glfw3.h>
#endif

ACMB_GUI_NAMESPACE_BEGIN

Window::Window( const std::string& name,  const ImVec2& size )
: _name( name )
, _size( size )
{

}

void Window::Show()
{
    ImGui::SetNextWindowPos( _pos, ImGuiCond_Always );
    ImGui::SetNextWindowSize( _size, ImGuiCond_Always );

    if ( !DrawHeader() )
        return ImGui::End();

    DrawDialog();
    ImGui::End();
}

bool Window::DrawHeader()
{
    return ImGui::Begin( _name.c_str(), nullptr, flags() );
}

void Window::SetPos( const ImVec2& pos )
{
    _pos = pos;
}

float Window::GetMenuScaling()
{
#ifdef _WIN32
    auto activeWindow = GetActiveWindow();
    HMONITOR monitor = MonitorFromWindow( activeWindow, MONITOR_DEFAULTTONEAREST );

    // Get the logical width and height of the monitor
    MONITORINFOEX monitorInfoEx;
    monitorInfoEx.cbSize = sizeof( monitorInfoEx );
    GetMonitorInfo( monitor, &monitorInfoEx );
    auto cxLogical = monitorInfoEx.rcMonitor.right - monitorInfoEx.rcMonitor.left;

    // Get the physical width and height of the monitor
    DEVMODE devMode;
    devMode.dmSize = sizeof( devMode );
    devMode.dmDriverExtra = 0;
    EnumDisplaySettings( monitorInfoEx.szDevice, ENUM_CURRENT_SETTINGS, &devMode );
    auto cxPhysical = devMode.dmPelsWidth;

    // Calculate the scaling factor
    return float( cxPhysical ) / cxLogical;
#elif defined ( __linux__ )
    static const auto glfwHandle = glfwInit();
    (void)glfwHandle;
    float xscale, yscale;
    glfwGetMonitorContentScale( glfwGetPrimaryMonitor(), &xscale, &yscale);
    return xscale;
#endif

    return 1.0f;
}

ACMB_GUI_NAMESPACE_END


