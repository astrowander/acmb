#include "../Tests/test.h"
#include "../GUI/TopToolbar.h"
#include "TestAppHost.h"
#include <map>

namespace
{
struct ToolbarFixture
{
    acmb::gui::ProcessingState processing;
    bool cudaAvailable = true;
    bool cuda = false;
    int openCalls = 0, saveCalls = 0, runCalls = 0, stopCalls = 0, helpCalls = 0;
    size_t lastPosition = 0;
    bool lastReplace = false;
    std::map<std::string, ImVec2> centers;
    acmb::gui::MenuItem open{"", "Open", "Open project", [this](size_t p, bool r) {
        ++openCalls; lastPosition = p; lastReplace = r;
    }};
    acmb::gui::MenuItem save{"", "Save", "Save project", [this](size_t p, bool r) {
        ++saveCalls; lastPosition = p; lastReplace = r;
    }};
    acmb::gui::MenuItem run{"", "Run", "Start processing", [this](size_t, bool) {
        ++runCalls; processing.Begin();
    }};
    acmb::gui::MenuItem stop{"", "Stop", "Stop processing", [this](size_t, bool) {
        ++stopCalls; processing.RequestStop();
    }, true};
    acmb::gui::MenuItem help{"", "Help", "Instructions", [this](size_t p, bool r) {
        ++helpCalls; lastPosition = p; lastReplace = r;
    }};

    void Draw()
    {
        centers.clear();
        ImGui::SetNextWindowPos({0, 0});
        ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
        ImGui::Begin("Toolbar fixture", nullptr, ImGuiWindowFlags_NoDecoration);
        const acmb::gui::ToolbarActions actions{&open, &save, &run, &stop, &help};
        acmb::gui::DrawTopToolbar({ImGui::GetContentRegionAvail().x, 60}, actions,
            [this] { return processing.IsBusy(); }, 3, true, cudaAvailable, cuda,
            [this](const char* id) {
                const auto min = ImGui::GetItemRectMin();
                const auto max = ImGui::GetItemRectMax();
                centers[id] = {(min.x + max.x) * 0.5f, (min.y + max.y) * 0.5f};
            });
        ImGui::End();
    }
};
}

ACMB_TESTS_NAMESPACE_BEGIN

BEGIN_SUITE(TopToolbar)

BEGIN_TEST(ProjectCallbacksAndStopRemainAvailableAtSmallSizes)
    for (const float scale : {1.0f, 1.5f})
    {
        TestAppHost host({{640, 480}, scale});
        ToolbarFixture fixture;
        const auto draw = [&] { fixture.Draw(); };
        host.RenderFrame(draw);
        host.RenderFrame(draw);
        for (const char* command : {"Open", "Save", "Run", "Stop"})
        {
            EXPECT_TRUE(fixture.centers.at(command).x > 0);
            EXPECT_TRUE(fixture.centers.at(command).x < 640);
        }
        host.Click(fixture.centers.at("Open"), draw);
        host.Click(fixture.centers.at("Save"), draw);
        EXPECT_EQ(1, fixture.openCalls);
        EXPECT_EQ(1, fixture.saveCalls);
        EXPECT_EQ(size_t(3), fixture.lastPosition);
        EXPECT_TRUE(fixture.lastReplace);

        host.Click(fixture.centers.at("Run"), draw);
        EXPECT_TRUE(fixture.processing.IsBusy());
        host.Click(fixture.centers.at("Open"), draw);
        host.Click(fixture.centers.at("Save"), draw);
        host.Click(fixture.centers.at("Run"), draw);
        EXPECT_EQ(1, fixture.openCalls);
        EXPECT_EQ(1, fixture.saveCalls);
        EXPECT_EQ(1, fixture.runCalls);
        host.Click(fixture.centers.at("Stop"), draw);
        EXPECT_EQ(1, fixture.stopCalls);
        EXPECT_TRUE(fixture.processing.IsCancellationRequested());
        EXPECT_TRUE(fixture.processing.IsBusy());
        fixture.processing.Finish();
        host.Click(fixture.centers.at("Save"), draw);
        EXPECT_EQ(2, fixture.saveCalls);
    }
END_TEST

BEGIN_TEST(SettingsAndHelpMenusDispatchExistingStateAndAction)
    TestAppHost host({{640, 480}});
    ToolbarFixture fixture;
    const auto draw = [&] { fixture.Draw(); };
    host.RenderFrame(draw);
    host.RenderFrame(draw);
    host.Click(fixture.centers.at("Settings"), draw);
    EXPECT_TRUE(fixture.centers.contains("Enable CUDA"));
    host.Click(fixture.centers.at("Enable CUDA"), draw);
    EXPECT_TRUE(fixture.cuda);
    host.Click({600, 350}, draw); // Dismiss any open menu before the next action.
    host.Click(fixture.centers.at("Help"), draw);
    EXPECT_TRUE(fixture.centers.contains("Instructions"));
    host.Click(fixture.centers.at("Instructions"), draw);
    EXPECT_EQ(1, fixture.helpCalls);
    EXPECT_EQ(size_t(3), fixture.lastPosition);
    EXPECT_TRUE(fixture.lastReplace);
END_TEST

BEGIN_TEST(SettingsRespectCudaAvailabilityAndProcessingLocks)
    TestAppHost host({{640, 480}});
    ToolbarFixture fixture;
    fixture.cudaAvailable = false;
    const auto draw = [&] { fixture.Draw(); };
    host.RenderFrame(draw);
    host.RenderFrame(draw);
    host.Click(fixture.centers.at("Settings"), draw);
    EXPECT_TRUE(fixture.centers.contains("Enable CUDA"));
    host.Click(fixture.centers.at("Enable CUDA"), draw);
    EXPECT_FALSE(fixture.cuda);
    host.Click({600, 350}, draw);
    fixture.processing.Begin();
    host.Click(fixture.centers.at("Settings"), draw);
    EXPECT_FALSE(fixture.centers.contains("Enable CUDA"));
    host.Click(fixture.centers.at("Help"), draw);
    EXPECT_FALSE(fixture.centers.contains("Instructions"));
    EXPECT_EQ(0, fixture.helpCalls);
    host.Click(fixture.centers.at("Stop"), draw);
    EXPECT_TRUE(fixture.processing.IsCancellationRequested());
END_TEST

END_SUITE

ACMB_TESTS_NAMESPACE_END
