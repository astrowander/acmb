#include "FontRegistry.h"
#include "imgui.h"
#include "window.h"
#include "./../Tools/SystemTools.h"

ACMB_GUI_NAMESPACE_BEGIN

FontRegistry::FontRegistry( float )
{
    ImGuiIO& io = ImGui::GetIO();
    const std::string acmb_path = GetEnv( "ACMB_PATH" );

    byDefault = io.Fonts->AddFontFromFileTTF( ( acmb_path + "/GUI/Fonts/NotoSans-Regular.ttf" ).c_str(), 15 );
    bold = io.Fonts->AddFontFromFileTTF( ( acmb_path + "/GUI/Fonts/NotoSans-Bold.ttf" ).c_str(), 15 );
    big = io.Fonts->AddFontFromFileTTF( ( acmb_path + "/GUI/Fonts/NotoSans-Regular.ttf" ).c_str(), 20 );
    bigBold = io.Fonts->AddFontFromFileTTF( ( acmb_path + "/GUI/Fonts/NotoSans-Bold.ttf" ).c_str(), 20 );

    ImFontConfig iconsConfig;
    iconsConfig.GlyphMinAdvanceX = 32;
    const ImWchar iconRanges[] = { 0xe005, 0xf8ff, 0 };
    icons = io.Fonts->AddFontFromFileTTF( ( acmb_path + "/GUI/Fonts/fa-solid-900.ttf" ).c_str(), 30, &iconsConfig, iconRanges );
    io.Fonts->Build();
}

FontRegistry& FontRegistry::Instance()
{
    static FontRegistry instance( Window::cMenuScaling );
    return instance;
}

ACMB_GUI_NAMESPACE_END
