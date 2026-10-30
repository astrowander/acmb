#include "../Tests/test.h"
#include "../GUI/AdaptivePanels.h"
#include "../GUI/ToolLibrary.h"
#include "TestAppHost.h"
#include <chrono>

namespace
{
struct TemporaryLayoutFile
{
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("acmb-gui-layout-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".ini");
    ~TemporaryLayoutFile()
    {
        std::error_code ignored;
        std::filesystem::remove(path, ignored);
        auto temporary = path; temporary += ".tmp";
        std::filesystem::remove(temporary, ignored);
    }
};

void BeginFixture()
{
    ImGui::SetNextWindowPos({0, 0});
    ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
    ImGui::Begin("Adaptive fixture", nullptr, ImGuiWindowFlags_NoDecoration);
}
}

ACMB_TESTS_NAMESPACE_BEGIN
BEGIN_SUITE(AdaptivePanels)

BEGIN_TEST(PreferencesRoundTripOverwriteAndReset)
    TemporaryLayoutFile file;
    gui::LayoutPreferences preferences{370.125f, 410.5f, 260.25f, true};
    EXPECT_EQ(true, preferences.Save(file.path));
    gui::LayoutPreferences loaded;
    EXPECT_EQ(true, loaded.Load(file.path));
    EXPECT_TRUE(loaded == preferences);
    preferences = {};
    EXPECT_EQ(true, preferences.Save(file.path));
    EXPECT_TRUE(loaded.Load(file.path));
    EXPECT_TRUE(loaded == gui::LayoutPreferences{});
END_TEST

BEGIN_TEST(MissingMalformedAndFutureFilesPreserveCurrentPreferences)
    TemporaryLayoutFile file;
    const gui::LayoutPreferences original{400, 450, 300, true};
    auto preferences = original;
    EXPECT_FALSE(preferences.Load(file.path));
    for (const char* content : {"", "version 2", "version 1\ntoolbox_width 310\ninspector_width wrong",
        "version 1\ntoolbox_width 310\ninspector_width 320\npipeline_height 250\npipeline_collapsed 5"})
    {
        { std::ofstream out(file.path); out << content; }
        EXPECT_FALSE(preferences.Load(file.path));
        EXPECT_TRUE(preferences == original);
    }
    // A file used as a parent directory is a deterministic write failure.
    EXPECT_FALSE(preferences.Save(file.path / "layout.ini"));
END_TEST

BEGIN_TEST(PreferencesRejectNonFiniteValuesAndClampLoadedSizes)
    gui::LayoutPreferences preferences{NAN, INFINITY, -INFINITY};
    preferences.Clamp();
    EXPECT_TRUE(preferences == gui::LayoutPreferences{});
    TemporaryLayoutFile file;
    { std::ofstream out(file.path); out << "version 1\ntoolbox_width -10\ninspector_width 50000\npipeline_height 1\npipeline_collapsed 1\n"; }
    EXPECT_TRUE(preferences.Load(file.path));
    EXPECT_EQ(220.0f, preferences.toolboxWidth);
    EXPECT_EQ(800.0f, preferences.inspectorWidth);
    EXPECT_EQ(160.0f, preferences.pipelineHeight);
    EXPECT_TRUE(preferences.pipelineCollapsed);
END_TEST

BEGIN_TEST(SplitterRangesProtectEveryPanelAndIgnoreHiddenPanels)
    for (const auto splitter : {gui::LayoutSplitter::Toolbox, gui::LayoutSplitter::Inspector, gui::LayoutSplitter::Pipeline})
        for (float delta : {-5000.0f, 5000.0f})
        {
            gui::LayoutPreferences preferences;
            auto layout = gui::MainWindowLayout::Calculate({1280, 720}, {8, 4}, {8, 8}, 19, preferences);
            gui::ResizeLayoutPanel(preferences, layout, splitter, delta);
            layout = gui::MainWindowLayout::Calculate({1280, 720}, {8, 4}, {8, 8}, 19, preferences);
            EXPECT_TRUE(layout.toolbox.width >= 220);
            EXPECT_TRUE(layout.frameControls.width >= 260);
            EXPECT_TRUE(layout.preview.width >= 320);
            EXPECT_TRUE(layout.preview.height >= 120);
            EXPECT_TRUE(layout.pipeline.height >= 160);
        }
    gui::LayoutPreferences preferences;
    const auto before = preferences;
    auto narrow = gui::MainWindowLayout::Calculate({600, 480}, {8, 4}, {8, 8}, 19, preferences);
    EXPECT_FALSE(gui::ResizeLayoutPanel(preferences, narrow, gui::LayoutSplitter::Toolbox, 100));
    EXPECT_FALSE(gui::ResizeLayoutPanel(preferences, narrow, gui::LayoutSplitter::Inspector, 100));
    EXPECT_FALSE(gui::ResizeLayoutPanel(preferences, narrow, gui::LayoutSplitter::Pipeline, NAN));
    EXPECT_TRUE(preferences == before);
    preferences.pipelineCollapsed = true;
    narrow = gui::MainWindowLayout::Calculate({600, 480}, {8, 4}, {8, 8}, 19, preferences);
    EXPECT_FALSE(gui::ResizeLayoutPanel(preferences, narrow, gui::LayoutSplitter::Pipeline, 100));
END_TEST

BEGIN_TEST(ActualMouseDragResizesAndReleasesSplitter)
    TestAppHost host;
    gui::LayoutPreferences preferences;
    ImVec2 handle{};
    bool released = false;
    const auto draw = [&] {
        BeginFixture();
        const auto layout = gui::MainWindowLayout::Calculate({1280, 720}, {8, 4}, {8, 8}, 19, preferences);
        handle = {layout.toolboxSplitter.x + 3, layout.toolboxSplitter.y + 30};
        const auto drag = gui::DrawLayoutSplitter("tools", layout.toolboxSplitter, true);
        gui::ResizeLayoutPanel(preferences, layout, gui::LayoutSplitter::Toolbox, drag.delta);
        released |= drag.released;
        ImGui::End();
    };
    host.RenderFrame(draw);
    auto& io = ImGui::GetIO();
    io.AddMousePosEvent(handle.x, handle.y); host.RenderFrame(draw);
    io.AddMouseButtonEvent(0, true); host.RenderFrame(draw);
    io.AddMousePosEvent(handle.x + 70, handle.y); host.RenderFrame(draw);
    io.AddMouseButtonEvent(0, false); host.RenderFrame(draw);
    EXPECT_EQ(390.0f, preferences.toolboxWidth);
    EXPECT_TRUE(released);
END_TEST

BEGIN_TEST(DrawerToolsKeepCallbacksLocksAndCloseAfterSelection)
    TestAppHost host({{600, 480}});
    gui::Items items;
    int actions = 0;
    size_t selectedPosition = 0;
    bool requested = true, open = false, locked = true, replaced = false;
    ImVec2 tool{};
    items["Tools"].emplace(1, std::make_unique<gui::MenuItem>("+", "Import", "",
        [&](size_t position, bool replace) { ++actions; selectedPosition = position; replaced = replace; }));
    const auto groups = gui::MenuItemsHolder::BuildToolGroups(items);
    const auto draw = [&] {
        BeginFixture();
        const auto layout = gui::MainWindowLayout::Calculate({600, 480}, {8, 4}, {8, 8}, 19);
        if (gui::BeginLayoutDrawer("Tools", "Tools", layout.toolsDrawer, {0, 0}, requested))
        {
            gui::DrawToolLibrary(groups, [&] { return locked; }, 3, true, {},
                [&](const char*, int id) {
                    if (id == 1) { const auto a = ImGui::GetItemRectMin(), b = ImGui::GetItemRectMax(); tool = {(a.x+b.x)/2, (a.y+b.y)/2}; }
                }, true);
            ImGui::EndPopup();
        }
        open = ImGui::IsPopupOpen("Tools");
        ImGui::End();
    };
    host.RenderFrame(draw); host.RenderFrame(draw);
    EXPECT_TRUE(open);
    host.Click(tool, draw);
    EXPECT_EQ(0, actions);
    EXPECT_TRUE(open);
    locked = false;
    host.Click(tool, draw);
    EXPECT_EQ(1, actions);
    EXPECT_EQ(size_t(3), selectedPosition);
    EXPECT_TRUE(replaced);
    EXPECT_FALSE(open);
END_TEST

BEGIN_TEST(DrawersStayInsideSmallViewportsAndCloseWhenDocked)
    for (ImVec2 size : {ImVec2{320, 240}, ImVec2{600, 480}, ImVec2{800, 600}})
    {
        const auto layout = gui::MainWindowLayout::Calculate(size, {8, 4}, {8, 8}, 19);
        for (const auto& drawer : {layout.toolsDrawer, layout.inspectorDrawer})
            if (drawer.width > 0)
            {
                EXPECT_TRUE(drawer.x >= 0 && drawer.y >= 0);
                EXPECT_TRUE(drawer.GetRight() <= size.x && drawer.GetBottom() <= size.y);
            }
    }
    TestAppHost host;
    bool requested = true, docked = false, open = false;
    const auto draw = [&] {
        BeginFixture();
        if (gui::BeginLayoutDrawer("Inspector", "Inspector", docked ? RectF{} : RectF{200, 70, 300, 400}, {0, 0}, requested))
            ImGui::EndPopup();
        open = ImGui::IsPopupOpen("Inspector");
        ImGui::End();
    };
    host.RenderFrame(draw); host.RenderFrame(draw);
    EXPECT_TRUE(open);
    docked = true;
    host.RenderFrame(draw);
    EXPECT_FALSE(open);
END_TEST

BEGIN_TEST(EveryDrawerControlRemainsReachableAcrossSizesAndScales)
    for (const auto size : {ImVec2{1280, 720}, ImVec2{800, 600}, ImVec2{640, 480}, ImVec2{600, 480}})
        for (float scale : {1.0f, 1.25f, 1.5f})
            for (bool inspector : {false, true})
            {
                TestAppHost host({size, scale});
                gui::MainWindowLayout layout;
                host.RenderFrame([&] {
                    const auto& style = ImGui::GetStyle();
                    layout = gui::MainWindowLayout::Calculate(size, style.ItemSpacing, style.WindowPadding,
                        ImGui::GetTextLineHeightWithSpacing());
                });
                const auto bounds = inspector ? layout.inspectorDrawer : layout.toolsDrawer;
                if (bounds.width <= 0) continue;
                bool requested = true;
                int reveal = -1, activated = -1;
                ImVec2 centers[20]{};
                bool visible[20]{};
                const auto draw = [&] {
                    BeginFixture();
                    if (gui::BeginLayoutDrawer("drawer", inspector ? "Inspector" : "Tools", bounds, {0, 0}, requested))
                    {
                        gui::BeginVerticalPanel("controls", {0, 0}, 220);
                        for (int control = 0; control < 20; ++control)
                        {
                            ImGui::PushID(control);
                            gui::IconTextButton("control", "+", "Control " + std::to_string(control),
                                ImGui::GetContentRegionAvail().x, false, [&] { activated = control; });
                            const auto a = ImGui::GetItemRectMin(), b = ImGui::GetItemRectMax();
                            centers[control] = {(a.x + b.x) / 2, (a.y + b.y) / 2};
                            const auto position = ImGui::GetWindowPos(), panelSize = ImGui::GetWindowSize();
                            const auto padding = ImGui::GetStyle().WindowPadding;
                            visible[control] = ImGui::IsItemVisible() && a.x >= 0 && b.x <= size.x &&
                                a.y >= position.y + padding.y - 0.5f && b.y <= position.y + panelSize.y - padding.y + 0.5f &&
                                a.y >= 0 && b.y <= size.y;
                            if (control == reveal) ImGui::SetScrollHereY(0.5f);
                            ImGui::PopID();
                        }
                        ImGui::EndChild();
                        ImGui::EndPopup();
                    }
                    ImGui::End();
                };
                host.RenderFrame(draw); host.RenderFrame(draw);
                for (int control = 0; control < 20; ++control)
                {
                    reveal = control;
                    host.RenderFrame(draw); host.RenderFrame(draw);
                    EXPECT_TRUE(visible[control]);
                    activated = -1;
                    host.Click(centers[control], draw);
                    EXPECT_EQ(control, activated);
                }
            }
END_TEST

END_SUITE
ACMB_TESTS_NAMESPACE_END
