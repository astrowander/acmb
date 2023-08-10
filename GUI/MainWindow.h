#pragma once
#include "window.h"
#include "./../Geometry/size.h"
#include "./../Geometry/point.h"
#include <array>
#include<unordered_map>

ACMB_GUI_NAMESPACE_BEGIN

class PipelineElementWindow;
class ImageWriterWindow;
struct FontRegistry;

class MainWindow : public Window
{
    std::unordered_map<size_t, std::weak_ptr<ImageWriterWindow>> _writers;
    std::vector<std::string> _errors;

    static constexpr Size cGridSize = { 26, 26 };
    std::array<std::shared_ptr< PipelineElementWindow>, cGridSize.width* cGridSize.height> _grid;

    Size _viewportSize;
    Point _viewportStart;
    Point _activeCell;
    

    bool _isBusy = false;
    bool _finished = false;

    MainWindow( const ImVec2& pos, const ImVec2& size, const FontRegistry& fontRegistry );
    virtual void DrawDialog() override;

    void ProcessKeyboardEvents();

    virtual ImGuiWindowFlags flags() override { return  ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNav; }

    template<class ElementType>
    std::string AddElementToGrid( const Point& pos );

    const FontRegistry& _fontRegistry;

public:

    virtual void Show() override;
    static std::shared_ptr<MainWindow> Create( const FontRegistry& fontRegistry );
};

ACMB_GUI_NAMESPACE_END