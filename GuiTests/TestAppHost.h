#pragma once

#include "imgui.h"
#include <cmath>
#include <functional>
#include <stdexcept>

namespace acmb::tests
{
// Sizes are ImGui logical units. Framebuffer density and UI scale are separate
// inputs: changing density must not change the available layout space.
struct LayoutInputs
{
    ImVec2 viewportSize{1280.0f, 720.0f};
    float uiScale = 1.0f;
    ImVec2 framebufferScale{1.0f, 1.0f};
};

class TestAppHost
{
    ImGuiContext* _previousContext;
    ImGuiContext* _context;

    static void ValidateSize(ImVec2 size)
    {
        if (!std::isfinite(size.x) || !std::isfinite(size.y) || size.x < 0 || size.y < 0)
            throw std::invalid_argument("Viewport dimensions must be finite and nonnegative");
    }

public:
    explicit TestAppHost(const LayoutInputs& inputs = {})
    {
        ValidateSize(inputs.viewportSize);
        if (!std::isfinite(inputs.uiScale) || inputs.uiScale <= 0 ||
            !std::isfinite(inputs.framebufferScale.x) || inputs.framebufferScale.x <= 0 ||
            !std::isfinite(inputs.framebufferScale.y) || inputs.framebufferScale.y <= 0)
            throw std::invalid_argument("UI and framebuffer scales must be finite and positive");

        _previousContext = ImGui::GetCurrentContext();
        _context = ImGui::CreateContext();
        ImGui::SetCurrentContext(_context);
        auto& io = ImGui::GetIO();
        io.IniFilename = nullptr;
        io.LogFilename = nullptr;
        io.DisplaySize = inputs.viewportSize;
        io.DisplayFramebufferScale = inputs.framebufferScale;
        io.DeltaTime = 1.0f / 60.0f;

        // Start from a fresh style and the embedded font on every run, without
        // querying monitors, loading host fonts, or accumulating scaled sizes.
        ImGui::GetStyle() = ImGuiStyle{};
        ImGui::StyleColorsDark();
        ImGui::GetStyle().ScaleAllSizes(inputs.uiScale);
        ImFontConfig fontConfig;
        // Match ImGui's pixel snapping explicitly so fractional scales have a
        // documented, deterministic font size (125% -> 16px, 150% -> 20px).
        fontConfig.SizePixels = std::round(13.0f * inputs.uiScale);
        io.Fonts->AddFontDefault(&fontConfig);
        unsigned char* pixels;
        int width, height;
        io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
    }

    ~TestAppHost()
    {
        ImGui::DestroyContext(_context);
        ImGui::SetCurrentContext(_previousContext);
    }

    TestAppHost(const TestAppHost&) = delete;
    TestAppHost& operator=(const TestAppHost&) = delete;

    void Resize(ImVec2 viewportSize)
    {
        ValidateSize(viewportSize);
        ImGui::SetCurrentContext(_context);
        ImGui::GetIO().DisplaySize = viewportSize;
    }

    void RenderFrame(const std::function<void()>& draw)
    {
        ImGui::SetCurrentContext(_context);
        ImGui::NewFrame();
        draw();
        ImGui::Render();
    }

    void Click(ImVec2 position, const std::function<void()>& draw)
    {
        ImGui::SetCurrentContext(_context);
        auto& io = ImGui::GetIO();
        io.AddMousePosEvent(position.x, position.y);
        RenderFrame(draw);
        io.AddMouseButtonEvent(ImGuiMouseButton_Left, true);
        RenderFrame(draw);
        io.AddMouseButtonEvent(ImGuiMouseButton_Left, false);
        RenderFrame(draw);
    }

    void PressKey(ImGuiKey key, const std::function<void()>& draw)
    {
        ImGui::SetCurrentContext(_context);
        ImGui::GetIO().AddKeyEvent(key, true);
        RenderFrame(draw);
        ImGui::GetIO().AddKeyEvent(key, false);
        RenderFrame(draw);
    }

    void Wheel(ImVec2 position, float horizontal, float vertical, const std::function<void()>& draw)
    {
        ImGui::SetCurrentContext(_context);
        ImGui::GetIO().AddMousePosEvent(position.x, position.y);
        RenderFrame(draw);
        ImGui::GetIO().AddMouseWheelEvent(horizontal, vertical);
        RenderFrame(draw);
        RenderFrame(draw);
    }
};
}
