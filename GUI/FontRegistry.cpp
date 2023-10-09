#include "FontRegistry.h"
#include "imgui.h"
#include "window.h"

ACMB_GUI_NAMESPACE_BEGIN

FontRegistry::FontRegistry( float scaling )
{
    ImGuiIO& io = ImGui::GetIO();
    byDefault = io.Fonts->AddFontFromFileTTF( "Fonts/NotoSans-Regular.ttf", 10 * scaling );
    bold = io.Fonts->AddFontFromFileTTF( "Fonts/NotoSans-Bold.ttf", 10 * scaling );


    ImFontConfig iconsConfig;
    iconsConfig.GlyphMinAdvanceX = 32;
    const ImWchar iconRanges[] = { 0xe005, 0xf8ff, 0 };
    icons = io.Fonts->AddFontFromFileTTF( "Fonts/fa-solid-900.ttf", 20 * scaling, &iconsConfig, iconRanges );
    io.Fonts->Build();
}

FontRegistry& FontRegistry::Instance()
{
    static FontRegistry instance( Window::cMenuScaling );
    return instance;
}

ACMB_GUI_NAMESPACE_END