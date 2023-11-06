#pragma once
#include "./../Core/macros.h"
#include "imgui.h"
#include <string>
#include <cfloat>
#include <map>

namespace IGFD
{
    class FileDialog;
}

ACMB_GUI_NAMESPACE_BEGIN

class FileDialog
{
    IGFD::FileDialog* _pImGuiFileDialog;

    FileDialog();

public:

    void OpenDialog( const std::string& vKey, const std::string& vTitle, const char* vFilters, const std::string& vFilePathName, const int& vCountSelectionMax );
    void Close();
    std::string GetFilePathName();
    std::string GetCurrentPath();

    bool Display( const std::string& vKey, ImGuiWindowFlags vFlags = ImGuiWindowFlags_NoCollapse, ImVec2 vMinSize = ImVec2( 0, 0 ), 
                  ImVec2 vMaxSize = ImVec2( FLT_MAX, FLT_MAX ) );
                  
    bool IsOk() const;

    std::map<std::string, std::string> GetSelection();

    static FileDialog& Instance();
};

ACMB_GUI_NAMESPACE_END