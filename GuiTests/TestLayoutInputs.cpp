#include "../Tests/test.h"
#include "TestAppHost.h"
#include <limits>

namespace
{
struct Measurements
{
    ImVec2 buttonSize;
    ImVec2 contentSize;
    float fontSize;
};

Measurements Measure(acmb::tests::TestAppHost& host)
{
    Measurements result{};
    // Warm up the window before sampling geometry.
    for (int frame = 0; frame < 2; ++frame)
    {
        host.RenderFrame([&] {
            ImGui::SetNextWindowPos(ImVec2(0, 0));
            ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
            ImGui::Begin("Layout fixture", nullptr,
                ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoSavedSettings);
            result.contentSize = ImGui::GetContentRegionAvail();
            result.fontSize = ImGui::GetFontSize();
            ImGui::Button("Run###RunCommand");
            result.buttonSize = ImGui::GetItemRectSize();
            ImGui::End();
        });
    }
    return result;
}
}

ACMB_TESTS_NAMESPACE_BEGIN

BEGIN_SUITE(LayoutInputs)

BEGIN_TEST(ExplicitViewportAndScaleMatrix)
    const ImVec2 sizes[] = {{1920, 1080}, {1366, 768}, {1280, 720}};
    const float scales[] = {1.0f, 1.25f, 1.5f};
    for (const auto size : sizes)
    {
        for (const auto scale : scales)
        {
            TestAppHost host({size, scale});
            const auto measured = Measure(host);
            const auto* drawData = ImGui::GetDrawData();
            EXPECT_EQ(size.x, drawData->DisplaySize.x);
            EXPECT_EQ(size.y, drawData->DisplaySize.y);
            EXPECT_EQ(std::round(13.0f * scale), measured.fontSize);
            EXPECT_EQ(1.0f / 60.0f, ImGui::GetIO().DeltaTime);
            EXPECT_TRUE(measured.contentSize.x > 0);
            EXPECT_TRUE(measured.contentSize.y > 0);
            EXPECT_TRUE(measured.buttonSize.x <= measured.contentSize.x);
            EXPECT_TRUE(measured.buttonSize.y <= measured.contentSize.y);
            EXPECT_TRUE(ImGui::GetIO().IniFilename == nullptr);
            EXPECT_TRUE(ImGui::GetIO().LogFilename == nullptr);
        }
    }
END_TEST

BEGIN_TEST(IdenticalInputsProduceIdenticalGeometry)
    Measurements first;
    {
        TestAppHost host({{1366, 768}, 1.25f});
        first = Measure(host);
    }
    TestAppHost host({{1366, 768}, 1.25f});
    const auto second = Measure(host);
    EXPECT_EQ(first.buttonSize.x, second.buttonSize.x);
    EXPECT_EQ(first.buttonSize.y, second.buttonSize.y);
    EXPECT_EQ(first.contentSize.x, second.contentSize.x);
    EXPECT_EQ(first.contentSize.y, second.contentSize.y);
    EXPECT_EQ(first.fontSize, second.fontSize);
END_TEST

BEGIN_TEST(UiScaleChangesControlGeometry)
    Measurements normal;
    {
        TestAppHost host;
        normal = Measure(host);
    }
    TestAppHost host({{1280, 720}, 1.5f});
    const auto scaled = Measure(host);
    EXPECT_TRUE(scaled.fontSize > normal.fontSize);
    EXPECT_TRUE(scaled.buttonSize.x > normal.buttonSize.x);
    EXPECT_TRUE(scaled.buttonSize.y > normal.buttonSize.y);
END_TEST

BEGIN_TEST(FramebufferDensityPreservesLogicalLayout)
    Measurements normal;
    {
        TestAppHost host;
        normal = Measure(host);
    }
    TestAppHost host({{1280, 720}, 1.0f, {2.0f, 2.0f}});
    const auto dense = Measure(host);
    const auto* drawData = ImGui::GetDrawData();
    EXPECT_EQ(2.0f, drawData->FramebufferScale.x);
    EXPECT_EQ(2.0f, drawData->FramebufferScale.y);
    EXPECT_EQ(normal.buttonSize.x, dense.buttonSize.x);
    EXPECT_EQ(normal.buttonSize.y, dense.buttonSize.y);
    EXPECT_EQ(normal.contentSize.x, dense.contentSize.x);
    EXPECT_EQ(normal.contentSize.y, dense.contentSize.y);
END_TEST

BEGIN_TEST(ResizeUpdatesLayoutWithoutAccumulatingScale)
    TestAppHost host({{1920, 1080}, 1.25f});
    const auto before = Measure(host);
    host.Resize({1280, 720});
    const auto resized = Measure(host);
    EXPECT_EQ(1280.0f, ImGui::GetDrawData()->DisplaySize.x);
    EXPECT_EQ(720.0f, ImGui::GetDrawData()->DisplaySize.y);
    EXPECT_TRUE(resized.contentSize.x < before.contentSize.x);
    EXPECT_TRUE(resized.contentSize.y < before.contentSize.y);
    EXPECT_EQ(before.buttonSize.x, resized.buttonSize.x);
    EXPECT_EQ(before.buttonSize.y, resized.buttonSize.y);
    host.Resize({1920, 1080});
    const auto restored = Measure(host);
    EXPECT_EQ(before.contentSize.x, restored.contentSize.x);
    EXPECT_EQ(before.contentSize.y, restored.contentSize.y);
END_TEST

BEGIN_TEST(ContextsRestorePreviousState)
    auto* previous = ImGui::GetCurrentContext();
    {
        TestAppHost outer;
        auto* outerContext = ImGui::GetCurrentContext();
        const auto before = Measure(outer);
        {
            TestAppHost inner({{800, 600}, 1.5f});
            Measure(inner);
        }
        EXPECT_TRUE(ImGui::GetCurrentContext() == outerContext);
        const auto after = Measure(outer);
        EXPECT_EQ(before.contentSize.x, after.contentSize.x);
        EXPECT_EQ(before.buttonSize.y, after.buttonSize.y);
        EXPECT_EQ(1.0f, ImGui::GetStyle().FontScaleMain);
    }
    EXPECT_TRUE(ImGui::GetCurrentContext() == previous);
END_TEST

BEGIN_TEST(InvalidInputsAreRejectedBeforeCreatingContext)
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float infinity = std::numeric_limits<float>::infinity();
    const LayoutInputs invalid[] = {
        {{-1, 720}}, {{1280, nan}}, {{infinity, 720}},
        {{1280, 720}, 0}, {{1280, 720}, -1}, {{1280, 720}, nan},
        {{1280, 720}, 1, {0, 1}}, {{1280, 720}, 1, {1, infinity}}
    };
    auto* previous = ImGui::GetCurrentContext();
    for (const auto& inputs : invalid)
    {
        bool rejected = false;
        try { TestAppHost host(inputs); }
        catch (const std::invalid_argument&) { rejected = true; }
        EXPECT_TRUE(rejected);
        EXPECT_TRUE(ImGui::GetCurrentContext() == previous);
    }
END_TEST

BEGIN_TEST(MinimizedViewportCanRecover)
    TestAppHost host;
    host.Resize({0, 0});
    host.RenderFrame([] {});
    EXPECT_EQ(0.0f, ImGui::GetDrawData()->DisplaySize.x);
    EXPECT_EQ(0.0f, ImGui::GetDrawData()->DisplaySize.y);
    host.Resize({1280, 720});
    const auto recovered = Measure(host);
    EXPECT_TRUE(recovered.contentSize.x > 0);
    EXPECT_TRUE(recovered.contentSize.y > 0);
END_TEST

END_SUITE

ACMB_TESTS_NAMESPACE_END
