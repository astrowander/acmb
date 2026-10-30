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

BEGIN_TEST(SharedProgressFitsBelowExpandedAndCollapsedPipeline)
    for (auto size : {ImVec2{1920, 1080}, ImVec2{1366, 768}, ImVec2{1280, 720}, ImVec2{640, 480}})
        for (float scale : {1.0f, 1.25f, 1.5f})
            for (bool collapsed : {false, true})
            {
                gui::LayoutPreferences preferences;
                preferences.pipelineCollapsed = collapsed;
                const auto layout = gui::MainWindowLayout::Calculate(size, {8*scale, 4*scale},
                    {8*scale, 8*scale}, 19*scale, preferences);
                EXPECT_TRUE(IsContained(layout.pipelineProgress, layout.canvasSize));
                EXPECT_TRUE(layout.pipeline.GetBottom() < layout.pipelineProgress.y);
                EXPECT_TRUE(layout.pipelineCommands.GetBottom() < layout.pipelineProgress.y);
                EXPECT_EQ(layout.toolbar.width, layout.pipelineProgress.width);
                EXPECT_TRUE(layout.preview.height >= gui::MainWindowLayout::MinimumPreviewHeight * layout.scale - 0.001f);
            }
END_TEST

BEGIN_TEST(DefaultDesktopUsesPreferredSizesAndMinimumPreview)
    const auto layout = gui::MainWindowLayout::Calculate({1280, 720}, {8, 4}, {8, 8}, 19);
    EXPECT_TRUE(layout.mode == gui::PanelMode::Wide);
    EXPECT_EQ(320.0f, layout.toolbox.width);
    EXPECT_EQ(340.0f, layout.frameControls.width);
    EXPECT_TRUE(layout.preview.width >= gui::MainWindowLayout::MinimumPreviewWidth);
    EXPECT_EQ(160.0f, layout.pipeline.height);
    EXPECT_EQ(4, layout.visibleCells);
    EXPECT_EQ(200.0f, layout.cardSize.x);
    EXPECT_EQ(250.0f, layout.cellWidth);
END_TEST

BEGIN_TEST(ModesChangeAtMinimumPanelWidths)
    const auto measure = [](float width) { return gui::MainWindowLayout::Calculate({width, 720}, {8, 4}, {8, 8}, 19); };
    EXPECT_TRUE(measure(617).mode == gui::PanelMode::Narrow);
    EXPECT_TRUE(measure(618).mode == gui::PanelMode::Medium);
    EXPECT_TRUE(measure(859).mode == gui::PanelMode::Medium);
    EXPECT_TRUE(measure(860).mode == gui::PanelMode::Wide);
    EXPECT_EQ(0.0f, measure(617).frameControls.width);
    EXPECT_TRUE(measure(617).inspectorDrawer.width >= 260);
    EXPECT_EQ(0.0f, measure(618).toolbox.width);
    EXPECT_TRUE(measure(618).toolsDrawer.width >= 220);
    EXPECT_TRUE(measure(860).preview.width >= 320);
END_TEST

BEGIN_TEST(PreviewFitsAfterPanelAndPipelineResizing)
    for (float width : {600.0f, 800.0f, 1280.0f})
    {
        const auto layout = gui::MainWindowLayout::Calculate({width, 720}, {8, 4}, {8, 8}, 19, {500, 500, 350});
        for (const auto image : {ImVec2{6000, 4000}, ImVec2{1000, 6000}, ImVec2{100, 80}})
        {
            const auto fitted = layout.FitPreview(image);
            EXPECT_TRUE(fitted.x >= layout.preview.x && fitted.y >= layout.preview.y);
            EXPECT_TRUE(fitted.GetRight() <= layout.preview.GetRight() + 0.001f);
            EXPECT_TRUE(fitted.GetBottom() <= layout.preview.GetBottom() + 0.001f);
            EXPECT_TRUE(std::abs(fitted.width / fitted.height - image.x / image.y) < 0.001f);
            EXPECT_TRUE(fitted.width <= image.x && fitted.height <= image.y);
        }
        EXPECT_EQ(0.0f, layout.FitPreview({0, 400}).width);
    }
END_TEST

BEGIN_TEST(PreferredSizesSurviveDrawersAndCollapse)
    gui::LayoutPreferences preferences{390, 420, 280, false};
    const auto measure = [&](float width) { return gui::MainWindowLayout::Calculate({width, 900}, {8, 4}, {8, 8}, 19, preferences); };
    const auto before = measure(1920);
    EXPECT_TRUE(measure(600).mode == gui::PanelMode::Narrow);
    EXPECT_EQ(before.toolbox.width, measure(1920).toolbox.width);
    EXPECT_EQ(before.frameControls.width, measure(1920).frameControls.width);
    preferences.pipelineCollapsed = true;
    const auto collapsed = measure(1920);
    EXPECT_EQ(64.0f, collapsed.pipeline.height);
    EXPECT_TRUE(collapsed.preview.height > before.preview.height);
    preferences.pipelineCollapsed = false;
    EXPECT_EQ(before.pipeline.height, measure(1920).pipeline.height);
END_TEST

BEGIN_TEST(ExtremePreferencesKeepDockedPanelsUsable)
    for (float width : {618.0f, 860.0f, 1280.0f, 1920.0f})
        for (const auto preferences : {gui::LayoutPreferences{-500, -500, -500}, gui::LayoutPreferences{5000, 5000, 5000}})
        {
            const auto layout = gui::MainWindowLayout::Calculate({width, 480}, {8, 4}, {8, 8}, 19, preferences);
            EXPECT_TRUE(layout.preview.width >= 320);
            EXPECT_TRUE(layout.preview.height >= 120);
            if (layout.toolbox.width > 0) EXPECT_TRUE(layout.toolbox.width >= 220);
            if (layout.frameControls.width > 0) EXPECT_TRUE(layout.frameControls.width >= 260);
            EXPECT_TRUE(layout.pipeline.height >= 160);
        }
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
