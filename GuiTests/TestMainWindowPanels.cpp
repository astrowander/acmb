#include "../Tests/test.h"
#include "../GUI/MainWindowLayout.h"
#include "TestAppHost.h"

namespace
{
bool IsContained(const acmb::RectF& rect, ImVec2 size)
{
    return std::isfinite(rect.x) && std::isfinite(rect.y) &&
        std::isfinite(rect.width) && std::isfinite(rect.height) &&
        rect.x >= 0 && rect.y >= 0 && rect.width >= 0 && rect.height >= 0 &&
        rect.GetRight() <= size.x + 0.001f && rect.GetBottom() <= size.y + 0.001f;
}

bool Overlaps(const acmb::RectF& a, const acmb::RectF& b)
{
    if (a.width == 0 || a.height == 0 || b.width == 0 || b.height == 0)
        return false;
    return a.x < b.GetRight() && b.x < a.GetRight() &&
        a.y < b.GetBottom() && b.y < a.GetBottom();
}

acmb::gui::MainWindowLayout MeasurePanels(acmb::tests::TestAppHost& host)
{
    acmb::gui::MainWindowLayout result{};
    host.RenderFrame([&] {
        // The same production calculator and ImGui metrics MainWindow uses.
        const auto& style = ImGui::GetStyle();
        result = acmb::gui::MainWindowLayout::Calculate(ImGui::GetIO().DisplaySize,
            style.ItemSpacing, style.WindowPadding, ImGui::GetTextLineHeightWithSpacing());
    });
    return result;
}
}

ACMB_TESTS_NAMESPACE_BEGIN

BEGIN_SUITE(MainWindowPanels)

BEGIN_TEST(PreservesExistingDesktopGeometry)
    // A fixed reference catches accidental changes to the current panel layout.
    const auto layout = gui::MainWindowLayout::Calculate({1280, 720}, {8, 4}, {8, 8}, 17);
    EXPECT_EQ(368.0f, layout.preview.x);
    EXPECT_EQ(48.0f, layout.preview.y);
    EXPECT_EQ(540.0f, layout.preview.width);
    EXPECT_EQ(444.0f, layout.preview.height);
    EXPECT_EQ(956.0f, layout.frameControls.x);
    EXPECT_EQ(308.0f, layout.frameControls.width);
    EXPECT_EQ(336.0f, layout.toolbox.width);
    EXPECT_EQ(500.0f, layout.pipelineHeader.y);
    EXPECT_EQ(540.0f, layout.pipeline.y);
    EXPECT_EQ(700.0f, layout.pipeline.GetBottom());
    EXPECT_EQ(4, layout.visibleCells);
    EXPECT_EQ(200.0f, layout.cardSize.x);
    EXPECT_EQ(250.0f, layout.cellWidth);
END_TEST

BEGIN_TEST(PanelsFitWithoutOverlapAcrossSizeAndScaleMatrix)
    const ImVec2 sizes[] = {{1920, 1080}, {1366, 768}, {1280, 720}, {800, 600}};
    for (auto size : sizes)
    {
        for (float scale : {1.0f, 1.25f, 1.5f})
        {
            std::cout << "Panel case: " << size.x << "x" << size.y << " @ " << scale << '\n';
            TestAppHost host({size, scale});
            const auto layout = MeasurePanels(host);
            const RectF panels[] = {layout.toolbar, layout.toolbox, layout.preview, layout.frameControls, layout.pipelineCommands,
                layout.pipelineHeader, layout.pipeline};
            for (const auto& panel : panels)
                EXPECT_TRUE(IsContained(panel, layout.canvasSize));
            for (size_t i = 0; i < 7; ++i)
                for (size_t j = i + 1; j < 7; ++j)
                    EXPECT_FALSE(Overlaps(panels[i], panels[j]));
            EXPECT_TRUE(layout.visibleCells >= 1);
            EXPECT_TRUE(layout.cellWidth >= gui::MainWindowLayout::CellMinWidth);
            EXPECT_TRUE(layout.cellWidth * layout.visibleCells <= layout.pipeline.width);
            EXPECT_TRUE(layout.cardSize.x >= gui::MainWindowLayout::CardWidth);
        }
    }
END_TEST

BEGIN_TEST(ToolboxStopsAbovePipeline)
    TestAppHost host({{1366, 768}, 1.5f});
    const auto layout = MeasurePanels(host);
    EXPECT_TRUE(layout.toolbox.GetBottom() < layout.pipelineHeader.y);
    EXPECT_TRUE(layout.preview.GetBottom() < layout.pipelineHeader.y);
    EXPECT_TRUE(layout.frameControls.GetBottom() < layout.pipelineHeader.y);
    EXPECT_TRUE(layout.toolbox.GetRight() <= layout.toolboxSeparatorX);
    EXPECT_TRUE(layout.toolboxSeparatorX <= layout.preview.x);
    EXPECT_TRUE(layout.preview.GetRight() <= layout.inspectorSeparatorX);
    EXPECT_TRUE(layout.inspectorSeparatorX <= layout.frameControls.x);
END_TEST

BEGIN_TEST(SmallAndMinimizedWindowsHaveFiniteClampedBounds)
    const ImVec2 sizes[] = {{640, 480}, {320, 240}, {200, 100}, {25, 25},
        {24, 24}, {1, 1}, {0, 0}, {0, 720}, {1280, 0}};
    for (const auto size : sizes)
    {
        TestAppHost host({size, 1.5f});
        const auto layout = MeasurePanels(host);
        for (const auto& panel : {layout.toolbar, layout.toolbox, layout.preview, layout.frameControls, layout.pipelineCommands,
            layout.pipelineHeader, layout.pipeline})
            EXPECT_TRUE(IsContained(panel, layout.canvasSize));
        EXPECT_TRUE(layout.visibleCells >= 1);
        EXPECT_TRUE(std::isfinite(layout.cellWidth));
        EXPECT_TRUE(layout.cellWidth >= 0);
        EXPECT_TRUE(layout.toolbox.GetRight() <= layout.frameControls.x);
    }
END_TEST

BEGIN_TEST(PipelineColumnThresholdsAvoidDivisionByZero)
    EXPECT_EQ(1, gui::MainWindowLayout::CountVisibleCells(0));
    EXPECT_EQ(1, gui::MainWindowLayout::CountVisibleCells(24));
    EXPECT_EQ(1, gui::MainWindowLayout::CountVisibleCells(425));
    EXPECT_EQ(1, gui::MainWindowLayout::CountVisibleCells(426));
    EXPECT_EQ(1, gui::MainWindowLayout::CountVisibleCells(675));
    EXPECT_EQ(2, gui::MainWindowLayout::CountVisibleCells(676));
    EXPECT_EQ(6, gui::MainWindowLayout::CountVisibleCells(1920));
END_TEST

BEGIN_TEST(ResizeAndRestorePreservesPanelGeometry)
    TestAppHost host({{1920, 1080}, 1.25f});
    const auto before = MeasurePanels(host);
    host.Resize({1280, 720});
    const auto resized = MeasurePanels(host);
    EXPECT_TRUE(resized.preview.width < before.preview.width);
    EXPECT_TRUE(resized.preview.height < before.preview.height);
    EXPECT_TRUE(resized.visibleCells < before.visibleCells);
    EXPECT_EQ(before.toolbox.width, resized.toolbox.width);
    EXPECT_EQ(before.frameControls.width, resized.frameControls.width);
    host.Resize({0, 0});
    MeasurePanels(host);
    host.Resize({1920, 1080});
    const auto restored = MeasurePanels(host);
    EXPECT_EQ(before.preview.x, restored.preview.x);
    EXPECT_EQ(before.preview.width, restored.preview.width);
    EXPECT_EQ(before.preview.height, restored.preview.height);
    EXPECT_EQ(before.pipeline.y, restored.pipeline.y);
    EXPECT_EQ(before.cellWidth, restored.cellWidth);
END_TEST

BEGIN_TEST(FramebufferDensityDoesNotMovePanels)
    gui::MainWindowLayout normal;
    {
        TestAppHost host({{1366, 768}, 1.25f});
        normal = MeasurePanels(host);
    }
    TestAppHost host({{1366, 768}, 1.25f, {2, 2}});
    const auto dense = MeasurePanels(host);
    EXPECT_EQ(normal.preview.x, dense.preview.x);
    EXPECT_EQ(normal.preview.width, dense.preview.width);
    EXPECT_EQ(normal.preview.height, dense.preview.height);
    EXPECT_EQ(normal.toolbox.width, dense.toolbox.width);
    EXPECT_EQ(normal.frameControls.x, dense.frameControls.x);
    EXPECT_EQ(normal.pipeline.y, dense.pipeline.y);
    EXPECT_EQ(normal.visibleCells, dense.visibleCells);
END_TEST

END_SUITE

ACMB_TESTS_NAMESPACE_END
