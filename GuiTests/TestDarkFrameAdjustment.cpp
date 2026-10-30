#include "../Tests/test.h"
#include "../GUI/DarkFrameAdjustment.h"
#include "../GUI/GuiInteractions.h"
#include "TestAppHost.h"
#include <memory>

namespace
{
// Substitute image processing only; mouse activation and the complete adjustment
// action (validation, errors, multiplier assignment, locking) are production code.
struct FrameResult
{
    std::string failure;
    explicit operator bool() const { return failure.empty(); }
    int value() const { return 1; }
    const std::string& error() const { return failure; }
};

struct Input
{
    int frames = 0;
    int reads = 0;
    std::string failure;
    int GetTaskCount() const { return frames; }
    FrameResult RunTaskAndReportProgress(size_t) { ++reads; return {failure}; }
};

struct AdjustmentFixture
{
    bool locked = false;
    int locks = 0, unlocks = 0;
    void LockInterface() { locked = true; ++locks; }
    void UnlockInterface() { locked = false; ++unlocks; }

    acmb::tests::TestAppHost host;
    std::shared_ptr<Input> input;
    int darkFrames = 0, darkReads = 0, calculations = 0;
    std::string darkFailure, calculationFailure, error;
    float multiplier = 1.25f;
    bool showError = false;
    ImVec2 buttonCenter{};

    void Draw()
    {
        ImGui::SetNextWindowPos({0, 0});
        ImGui::SetNextWindowSize({600, 400});
        ImGui::Begin("Dark frame adjustment", nullptr,
            ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoSavedSettings);
        acmb::gui::CommandButton("Adjust Multiplier", {-1, 0}, locked, [&] {
            acmb::gui::AdjustDarkFrameMultiplier(*this, input, darkFrames,
                [&] { ++darkReads; return FrameResult{darkFailure}; },
                [&](int, int) {
                    ++calculations;
                    if (!calculationFailure.empty())
                        throw std::runtime_error(calculationFailure);
                    return 2.0f;
                }, multiplier, error, showError);
        });
        const auto min = ImGui::GetItemRectMin();
        const auto max = ImGui::GetItemRectMax();
        buttonCenter = {(min.x + max.x) * 0.5f, (min.y + max.y) * 0.5f};
        ImGui::End();
    }

    void Click()
    {
        const auto draw = [&] { Draw(); };
        host.RenderFrame(draw);
        host.RenderFrame(draw);
        host.Click(buttonCenter, draw);
    }
};
}

ACMB_TESTS_NAMESPACE_BEGIN

BEGIN_SUITE(DarkFrameAdjustment)

BEGIN_TEST(ClickWithoutInputReportsErrorAndUnlocks)
    for (bool connected : {false, true})
    {
        AdjustmentFixture fixture;
        if (connected)
            fixture.input = std::make_shared<Input>(); // Connected import with no frames.
        fixture.Click();
        EXPECT_TRUE(fixture.showError);
        EXPECT_EQ(std::string("No input frames for adjusting the dark frame multiplier"), fixture.error);
        EXPECT_EQ(1.25f, fixture.multiplier);
        EXPECT_EQ(0, fixture.darkReads);
        EXPECT_EQ(0, fixture.calculations);
        EXPECT_FALSE(fixture.locked);
        EXPECT_EQ(1, fixture.locks);
        EXPECT_EQ(1, fixture.unlocks);
        if (connected)
        {
            EXPECT_EQ(0, fixture.input->reads);
        }
    }
END_TEST

BEGIN_TEST(ClickWithoutDarkFramesReportsErrorAndUnlocks)
    AdjustmentFixture fixture;
    fixture.input = std::make_shared<Input>();
    fixture.input->frames = 1;
    fixture.Click();
    EXPECT_TRUE(fixture.showError);
    EXPECT_EQ(std::string("No dark frames for adjusting the dark frame multiplier"), fixture.error);
    EXPECT_EQ(1.25f, fixture.multiplier);
    EXPECT_EQ(0, fixture.input->reads);
    EXPECT_EQ(0, fixture.darkReads);
    EXPECT_EQ(0, fixture.calculations);
    EXPECT_FALSE(fixture.locked);
END_TEST

BEGIN_TEST(FrameAndCalculationErrorsPreserveMultiplierAndUnlock)
    for (int failureStage = 0; failureStage < 3; ++failureStage)
    {
        AdjustmentFixture fixture;
        fixture.input = std::make_shared<Input>();
        fixture.input->frames = fixture.darkFrames = 1;
        if (failureStage == 0) fixture.input->failure = "Cannot read light frame";
        if (failureStage == 1) fixture.darkFailure = "Cannot read dark frame";
        if (failureStage == 2) fixture.calculationFailure = "Incompatible frames";
        fixture.Click();
        EXPECT_TRUE(fixture.showError);
        const auto expectedError = failureStage == 0 ? fixture.input->failure :
            failureStage == 1 ? fixture.darkFailure : fixture.calculationFailure;
        EXPECT_EQ(expectedError, fixture.error);
        EXPECT_EQ(1.25f, fixture.multiplier);
        EXPECT_FALSE(fixture.locked);
        EXPECT_EQ(1, fixture.unlocks);
        EXPECT_EQ((failureStage == 0 ? 0 : 1), fixture.darkReads);
        EXPECT_EQ((failureStage == 2 ? 1 : 0), fixture.calculations);
    }
END_TEST

BEGIN_TEST(AdjustmentWorksAfterAddingMissingFrames)
    AdjustmentFixture fixture;
    fixture.Click();
    EXPECT_TRUE(fixture.showError);
    fixture.showError = false; // Dismiss the error dialog.
    fixture.input = std::make_shared<Input>();
    fixture.input->frames = fixture.darkFrames = 1;
    fixture.Click();
    EXPECT_FALSE(fixture.showError);
    EXPECT_EQ(2.0f, fixture.multiplier);
    EXPECT_EQ(1, fixture.input->reads);
    EXPECT_EQ(1, fixture.darkReads);
    EXPECT_EQ(1, fixture.calculations);
    EXPECT_FALSE(fixture.locked);
    EXPECT_EQ(2, fixture.unlocks);
END_TEST

END_SUITE

ACMB_TESTS_NAMESPACE_END
