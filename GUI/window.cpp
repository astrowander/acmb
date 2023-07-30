#include "window.h"

#ifdef _WIN32
#include <Windows.h>
#endif

ACMB_GUI_NAMESPACE_BEGIN

Window::Window( const std::string& name, const ImVec2& pos, const ImVec2& size, std::shared_ptr<Window> pParent )
: _name( name )
, _pos( pos )
, _size( size )
, _pParent( pParent )
{

}

void Window::Show( ImGuiWindowFlags flags )
{
    ImGui::SetNextWindowPos( _pos, ImGuiCond_Appearing );
    ImGui::SetNextWindowSize( _size, ImGuiCond_Appearing );

    if ( !ImGui::Begin( _name.c_str(), &_isOpen, flags ) )
        return ImGui::End();

    DrawDialog();
    ImGui::End();

    for ( auto child : _children )
        child->Show();
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


