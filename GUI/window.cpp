#include "window.h"

#ifdef _WIN32
#include <Windows.h>
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

    if ( !ImGui::Begin( _name.c_str(), nullptr, flags() ) )
        return ImGui::End();

    DrawDialog();
    ImGui::End();
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
#endif

}

ACMB_GUI_NAMESPACE_END


