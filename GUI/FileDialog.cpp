#include "FileDialog.h"
#include "MainWindow.h"
#include "ImGuiFileDialog/ImGuiFileDialog.h"
ACMB_GUI_NAMESPACE_BEGIN

FileDialog& FileDialog::Instance()
{
    static FileDialog instance;
    return instance;
}

FileDialog::FileDialog()
{
    _pImGuiFileDialog = ImGuiFileDialog::Instance();
}

void FileDialog::OpenDialog( const std::string& vKey, const std::string& vTitle, const char* vFilters, const std::string& vFilePathName, const int& vCountSelectionMax )
{
    _pImGuiFileDialog->OpenDialog( vKey, vTitle, vFilters, vFilePathName, vCountSelectionMax );
    MainWindow::GetInstance( FontRegistry::Instance() ).LockInterface();
}

void FileDialog::Close()
{
    _pImGuiFileDialog->Close();
    MainWindow::GetInstance( FontRegistry::Instance() ).UnlockInterface();
}

bool FileDialog::Display( const std::string& vKey, ImGuiWindowFlags vFlags, ImVec2 vMinSize, ImVec2 vMaxSize )
{
    return _pImGuiFileDialog->Display( vKey, vFlags, vMinSize, vMaxSize );    
}

bool FileDialog::IsOk() const
{
    return _pImGuiFileDialog->IsOk();
}

std::string FileDialog::GetFilePathName()
{
    return _pImGuiFileDialog->GetFilePathName();
}

std::map<std::string, std::string> FileDialog::GetSelection()
{
    return _pImGuiFileDialog->GetSelection();
}

std::string FileDialog::GetCurrentPath()
{
    return _pImGuiFileDialog->GetCurrentPath();
}

ACMB_GUI_NAMESPACE_END
