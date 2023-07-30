#pragma once
#include "window.h"


ACMB_GUI_NAMESPACE_BEGIN

class ImageWriterWindow;

class MainWindow : public Window
{
    std::vector<std::weak_ptr<ImageWriterWindow>> _writers;
    std::vector<std::string> _errors;

    bool _finished = false;

    MainWindow( const ImVec2& pos, const ImVec2& size );
    virtual void DrawDialog() override;

public:

    static std::shared_ptr<MainWindow> Create();
};

ACMB_GUI_NAMESPACE_END