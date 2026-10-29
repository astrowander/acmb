#include "../Tests/test.h"
#include "../GUI/ScrollablePanels.h"
#include "../GUI/GuiInteractions.h"
#include "TestAppHost.h"

namespace
{
void BeginFixture()
{
    ImGui::SetNextWindowPos({0, 0});
    ImGui::SetNextWindowSize({900, 650});
    ImGui::Begin("Scrolling fixture", nullptr, ImGuiWindowFlags_NoDecoration);
}

ImVec2 ItemCenter()
{
    auto min = ImGui::GetItemRectMin();
    auto max = ImGui::GetItemRectMax();
    return {(min.x + max.x) / 2, (min.y + max.y) / 2};
}
}

ACMB_TESTS_NAMESPACE_BEGIN

BEGIN_SUITE(ScrollablePanels)

BEGIN_TEST(WorkspacePreservesMinimumPanelAndCardSizes)
    const auto small = gui::MainWindowLayout::Calculate({640, 480}, {8, 4}, {8, 8}, 19);
    const auto large = gui::MainWindowLayout::Calculate({1920, 1080}, {8, 4}, {8, 8}, 19);
    EXPECT_EQ(1000.0f, small.canvasSize.x);
    EXPECT_EQ(600.0f, small.canvasSize.y);
    EXPECT_EQ(large.toolbox.width, small.toolbox.width);
    EXPECT_EQ(large.frameControls.width, small.frameControls.width);
    EXPECT_EQ(large.cardSize.x, small.cardSize.x);
    EXPECT_EQ(large.cellWidth, small.cellWidth);
    EXPECT_TRUE(small.preview.width >= 240);
    EXPECT_TRUE(small.toolbox.height >= 200);
    EXPECT_TRUE(small.pipeline.height >= 300);
END_TEST

BEGIN_TEST(IconTextRowsDoNotShrinkOrOverlap)
    for (float scale : {1.0f, 1.25f, 1.5f})
    {
        TestAppHost host({{1280, 720}, scale});
        ImVec2 sizes[3]{};
        ImVec2 positions[3]{};
        const auto draw = [&] {
            BeginFixture();
            gui::BeginVerticalPanel("tools", {330, 300}, 220);
            const char* labels[] = {"Open", "Dark frame", "An intentionally long caption to verify minimum width"};
            for (int i = 0; i < 3; ++i)
            {
                ImGui::PushID(i);
                gui::IconTextButton("tool", "+", labels[i], 32, false, [] {});
                sizes[i] = ImGui::GetItemRectSize();
                positions[i] = ImGui::GetItemRectMin();
                EXPECT_TRUE(sizes[i].x >= ImGui::CalcTextSize(labels[i]).x);
                EXPECT_TRUE(sizes[i].y >= 32);
                ImGui::PopID();
            }
            ImGui::EndChild();
            ImGui::End();
        };
        host.RenderFrame(draw);
        host.RenderFrame(draw);
        EXPECT_TRUE(sizes[2].x > sizes[0].x);
        EXPECT_TRUE(positions[1].y >= positions[0].y + sizes[0].y);
        EXPECT_TRUE(positions[2].y >= positions[1].y + sizes[1].y);
    }
END_TEST

BEGIN_TEST(SettingsControlsPreserveThemeSizes)
    TestAppHost host;
    ImVec2 fieldSize{}, buttonSize{};
    float value = 1;
    float available = 0;
    float frameHeight = 0;
    const auto draw = [&] {
        BeginFixture();
        frameHeight = ImGui::GetFrameHeight();
        gui::BeginVerticalPanel("settings", {320, 180}, 220);
        ImGui::SetNextItemWidth(150);
        ImGui::InputFloat("##setting", &value);
        fieldSize = ImGui::GetItemRectSize();
        available = ImGui::GetContentRegionAvail().x;
        gui::CommandButton("Apply", {-1, 0}, false, [] {});
        buttonSize = ImGui::GetItemRectSize();
        ImGui::EndChild();
        ImGui::End();
    };
    host.RenderFrame(draw);
    host.RenderFrame(draw);
    EXPECT_TRUE(fieldSize.x >= 150);
    EXPECT_EQ(frameHeight, fieldSize.y);
    EXPECT_EQ(frameHeight, buttonSize.y);
    EXPECT_EQ(available - 1, buttonSize.x);
END_TEST

BEGIN_TEST(SmallWorkspaceScrollsInsteadOfCompressingPanels)
    TestAppHost host({{640, 480}});
    float maxX = 0, maxY = 0;
    const auto draw = [&] {
        const auto& style = ImGui::GetStyle();
        const auto layout = gui::MainWindowLayout::Calculate({640, 480}, style.ItemSpacing,
            style.WindowPadding, ImGui::GetTextLineHeightWithSpacing());
        ImGui::SetNextWindowPos({0, 0});
        ImGui::SetNextWindowSize({640, 480});
        ImGui::SetNextWindowContentSize({layout.canvasSize.x - 2 * style.WindowPadding.x,
            layout.canvasSize.y - 2 * style.WindowPadding.y});
        ImGui::Begin("Small workspace", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_HorizontalScrollbar);
        maxX = ImGui::GetScrollMaxX();
        maxY = ImGui::GetScrollMaxY();
        ImGui::End();
    };
    host.RenderFrame(draw);
    host.RenderFrame(draw);
    EXPECT_TRUE(maxX > 0);
    EXPECT_TRUE(maxY > 0);
END_TEST

BEGIN_TEST(EveryToolAndSettingControlRemainsReachableAtSmallSizes)
    const ImVec2 sizes[] = {{1280, 720}, {1024, 600}, {800, 600}, {640, 480}};
    for (const auto size : sizes)
    for (const float scale : {1.0f, 1.25f, 1.5f})
    for (const char* panel : {"ToolboxSection", "SettingsSection"})
    {
        std::cout << panel << ": " << size.x << "x" << size.y << " @ " << scale << '\n';
        TestAppHost host({size, scale});
        float maxScroll = 0;
        float currentScroll = 0;
        int revealControl = -1;
        ImVec2 centers[20]{};
        bool visible[20]{};
        int activated = -1;
        const auto draw = [&] {
            BeginFixture();
            gui::BeginVerticalPanel(panel, {320, 180}, 220);
            maxScroll = ImGui::GetScrollMaxY();
            currentScroll = ImGui::GetScrollY();
            for (int i = 0; i < 20; ++i)
            {
                ImGui::PushID(i);
                gui::IconTextButton("row", "+", "Tool or setting " + std::to_string(i),
                    240, false, [&] { activated = i; });
                centers[i] = ItemCenter();
                const auto itemMin = ImGui::GetItemRectMin();
                const auto itemMax = ImGui::GetItemRectMax();
                const auto panelPosition = ImGui::GetWindowPos();
                const auto panelSize = ImGui::GetWindowSize();
                const auto padding = ImGui::GetStyle().WindowPadding;
                const float scrollbar = ImGui::GetStyle().ScrollbarSize;
                visible[i] = ImGui::IsItemVisible() &&
                    itemMin.x >= panelPosition.x + padding.x - 0.5f &&
                    itemMin.y >= panelPosition.y + padding.y - 0.5f &&
                    itemMax.x <= panelPosition.x + panelSize.x - padding.x - scrollbar + 0.5f &&
                    itemMax.y <= panelPosition.y + panelSize.y - padding.y + 0.5f &&
                    itemMin.x >= 0 && itemMin.y >= 0 && itemMax.x <= size.x && itemMax.y <= size.y;
                if (i == revealControl)
                    ImGui::SetScrollHereY(0.5f);
                ImGui::PopID();
            }
            ImGui::EndChild();
            ImGui::End();
        };
        host.RenderFrame(draw);
        host.RenderFrame(draw);
        EXPECT_TRUE(maxScroll > 0);
        host.Wheel({120, 100}, 0, -1, draw);
        EXPECT_TRUE(currentScroll > 0);
        // Check every control, not just the existence of a scrollbar or the
        // final row. Activate only through injected input after revealing it.
        for (int control = 0; control < 20; ++control)
        {
            revealControl = control;
            host.RenderFrame(draw);
            host.RenderFrame(draw);
            EXPECT_TRUE(visible[control]);
            EXPECT_TRUE(centers[control].x >= 0 && centers[control].x < size.x);
            EXPECT_TRUE(centers[control].y >= 0 && centers[control].y < size.y);
            activated = -1;
            host.Click(centers[control], draw);
            EXPECT_EQ(control, activated);
        }
    }
END_TEST

BEGIN_TEST(HorizontalScrollKeepsCardWidthAndSelectionCorrect)
    TestAppHost host;
    gui::PipelineSelection selection;
    float viewportWidth = 650;
    float maxScroll = 0;
    float scroll = 0;
    bool scrollToEnd = false;
    ImVec2 lastCenter{};
    ImVec2 cardSize{};
    const auto draw = [&] {
        BeginFixture();
        gui::BeginPipelinePanel("pipeline", {viewportWidth, 300}, 8, gui::MainWindowLayout::CellMinWidth);
        const auto& style = ImGui::GetStyle();
        if (scrollToEnd)
            ImGui::SetScrollX(ImGui::GetScrollMaxX());
        maxScroll = ImGui::GetScrollMaxX();
        scroll = ImGui::GetScrollX();
        auto hit = gui::MainWindowLayout::Calculate({1000, 650}, {8, 4}, {8, 8}, 19);
        hit.pipeline = {0, 0, viewportWidth - 2 * style.WindowPadding.x, 250};
        const auto position = ImGui::GetWindowPos();
        gui::ProcessSelectionMouse(selection, 8, hit,
            {position.x + style.WindowPadding.x, position.y + style.WindowPadding.y}, scroll, 0);
        for (int i = 0; i < 8; ++i)
        {
            if (i > 0)
                ImGui::SameLine(0, 50);
            ImGui::PushID(i);
            ImGui::BeginGroup();
            ImGui::BeginChild("card", {gui::MainWindowLayout::CardWidth, 220}, ImGuiChildFlags_Borders);
            gui::CommandButton("Select", {200, 32}, false, [] {});
            if (i == 7)
                lastCenter = ItemCenter();
            ImGui::EndChild();
            cardSize = ImGui::GetItemRectSize();
            ImGui::EndGroup();
            ImGui::PopID();
        }
        ImGui::EndChild();
        ImGui::End();
    };
    host.RenderFrame(draw);
    host.RenderFrame(draw);
    EXPECT_TRUE(maxScroll > 0);
    host.Wheel({120, 100}, -1, 0, draw);
    EXPECT_TRUE(scroll > 0);
    viewportWidth = 360;
    scrollToEnd = true;
    host.RenderFrame(draw);
    host.RenderFrame(draw);
    scrollToEnd = false;
    EXPECT_EQ(gui::MainWindowLayout::CardWidth, cardSize.x);
    EXPECT_EQ(220.0f, cardSize.y);
    host.Click(lastCenter, draw);
    host.Click(lastCenter, draw);
    EXPECT_EQ(7, selection.activeIndex);
    EXPECT_TRUE(selection.elementSelected);
END_TEST

END_SUITE

ACMB_TESTS_NAMESPACE_END
