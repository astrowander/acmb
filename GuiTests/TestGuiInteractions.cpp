#include "../Tests/test.h"
#include "../GUI/GuiInteractions.h"
#include "../Core/enums.h"
#include "TestAppHost.h"
#include <future>

namespace
{
ImVec2 LastItemCenter()
{
    const auto min = ImGui::GetItemRectMin();
    const auto max = ImGui::GetItemRectMax();
    return {(min.x + max.x) * 0.5f, (min.y + max.y) * 0.5f};
}

void BeginFixture(const char* name, ImVec2 position = {0, 0}, ImVec2 size = {600, 400})
{
    ImGui::SetNextWindowPos(position);
    ImGui::SetNextWindowSize(size);
    ImGui::Begin(name, nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoSavedSettings);
}
}

ACMB_TESTS_NAMESPACE_BEGIN

BEGIN_SUITE(GuiInteractions)

BEGIN_TEST(MouseSelectionAndKeyboardNavigation)
    TestAppHost host;
    gui::PipelineSelection selection;
    gui::MainWindowLayout layout;
    const ImVec2 origin{40, 25};
    const auto draw = [&] {
        BeginFixture("Pipeline interaction", origin, {1000, 650});
        const auto& style = ImGui::GetStyle();
        layout = gui::MainWindowLayout::Calculate({1000, 650}, style.ItemSpacing,
            style.WindowPadding, ImGui::GetTextLineHeightWithSpacing());
        gui::ProcessSelectionMouse(selection, 3, layout, origin);
        gui::ProcessSelectionKeys(selection, 3, layout.visibleCells);
        ImGui::End();
    };
    host.RenderFrame(draw);
    host.RenderFrame(draw);
    EXPECT_EQ(0, selection.activeIndex);

    // Use production panel geometry rather than hardcoded screen coordinates.
    const ImVec2 target{origin.x + layout.pipeline.x + layout.cellWidth * 2 +
        (layout.cellWidth - layout.cardSize.x) * 0.5f + 20,
        origin.y + layout.pipeline.y + layout.pipeline.height * 0.5f};
    host.Click(target, draw);
    EXPECT_EQ(0, selection.activeIndex); // A single click must not select.
    host.Click(target, draw);
    EXPECT_EQ(2, selection.activeIndex);
    EXPECT_TRUE(selection.elementSelected);

    host.PressKey(ImGuiKey_LeftArrow, draw);
    EXPECT_EQ(2, selection.activeIndex);
    EXPECT_FALSE(selection.elementSelected);
    host.PressKey(ImGuiKey_RightArrow, draw);
    EXPECT_EQ(2, selection.activeIndex);
    EXPECT_TRUE(selection.elementSelected);
    host.PressKey(ImGuiKey_RightArrow, draw);
    EXPECT_EQ(3, selection.activeIndex);
    EXPECT_FALSE(selection.elementSelected);
END_TEST

BEGIN_TEST(SettingClickChangesOnlySelectedStepAndHonorsLock)
    TestAppHost host;
    gui::PipelineSelection selection;
    int modes[] = {int(StackMode::Light), int(StackMode::Light)};
    bool locked = false;
    int changes = 0;
    ImVec2 lightCenter{}, simpleCenter{};
    const auto draw = [&] {
        BeginFixture("Settings interaction");
        ImGui::PushID(selection.activeIndex);
        int& value = modes[selection.activeIndex];
        if (gui::RadioSetting("Light Frames", value, int(StackMode::Light), locked))
            ++changes;
        lightCenter = LastItemCenter();
        if (gui::RadioSetting("Simple", value, int(StackMode::Simple), locked))
            ++changes;
        simpleCenter = LastItemCenter();
        ImGui::PopID();
        ImGui::End();
    };
    host.RenderFrame(draw);
    host.RenderFrame(draw);
    host.Click(simpleCenter, draw);
    EXPECT_EQ(int(StackMode::Simple), modes[0]);
    EXPECT_EQ(int(StackMode::Light), modes[1]);
    EXPECT_EQ(1, changes);

    selection.Select(1, true, 2, 3);
    host.RenderFrame(draw);
    host.Click(simpleCenter, draw);
    EXPECT_EQ(int(StackMode::Simple), modes[0]);
    EXPECT_EQ(int(StackMode::Simple), modes[1]);
    EXPECT_EQ(2, changes);

    locked = true;
    host.Click(lightCenter, draw);
    EXPECT_EQ(int(StackMode::Simple), modes[1]);
    EXPECT_EQ(2, changes);
    locked = false;
    host.Click(lightCenter, draw);
    EXPECT_EQ(int(StackMode::Light), modes[1]);
    EXPECT_EQ(int(StackMode::Simple), modes[0]);
    EXPECT_EQ(3, changes);
END_TEST

BEGIN_TEST(RunStopBlocksEditsUntilWorkerCompletion)
    TestAppHost host;
    gui::ProcessingState state;
    std::promise<void> completeWorker;
    auto completion = completeWorker.get_future();
    int starts = 0;
    int edits = 0;
    ImVec2 runCenter{}, stopCenter{}, editCenter{};
    const auto draw = [&] {
        if (state.IsBusy() && completion.valid() &&
            completion.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
        {
            completion.get();
            state.Finish();
        }
        BeginFixture("Processing interaction");
        gui::CommandButton("Run###RunCommand", {100, 30}, state.IsBusy(), [&] {
            if (state.Begin())
                ++starts;
        });
        runCenter = LastItemCenter();
        ImGui::SameLine();
        // Like MainWindow's unlockable Stop button, this remains actionable
        // while processing locks other commands.
        gui::CommandButton("Stop###StopCommand", {100, 30}, false,
            [&] { state.RequestStop(); });
        stopCenter = LastItemCenter();
        gui::CommandButton("Edit###EditCommand", {100, 30}, state.IsBusy(), [&] { ++edits; });
        editCenter = LastItemCenter();
        ImGui::End();
    };
    host.RenderFrame(draw);
    host.RenderFrame(draw);
    host.Click(stopCenter, draw);
    EXPECT_FALSE(state.IsBusy());
    EXPECT_FALSE(state.IsCancellationRequested());

    host.Click(runCenter, draw);
    EXPECT_TRUE(state.IsBusy());
    EXPECT_FALSE(state.IsCancellationRequested());
    EXPECT_EQ(1, starts);
    host.Click(runCenter, draw);
    host.Click(editCenter, draw);
    EXPECT_EQ(1, starts);
    EXPECT_EQ(0, edits);

    host.Click(stopCenter, draw);
    EXPECT_TRUE(state.IsCancellationRequested());
    EXPECT_TRUE(state.IsBusy()); // Cancellation is not worker completion.
    host.Click(editCenter, draw);
    EXPECT_EQ(0, edits);

    completeWorker.set_value();
    host.RenderFrame(draw);
    EXPECT_FALSE(state.IsBusy());
    host.Click(editCenter, draw);
    EXPECT_EQ(1, edits);

    completeWorker = std::promise<void>{};
    completion = completeWorker.get_future();
    host.Click(runCenter, draw);
    EXPECT_EQ(2, starts);
    EXPECT_TRUE(state.IsBusy());
    EXPECT_FALSE(state.IsCancellationRequested());
    completeWorker.set_value();
    host.RenderFrame(draw);
    EXPECT_FALSE(state.IsBusy());
END_TEST

END_SUITE

ACMB_TESTS_NAMESPACE_END
