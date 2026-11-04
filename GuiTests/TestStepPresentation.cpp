#include "../Tests/test.h"
#include "../GUI/StepPresentation.h"
#include "../GUI/GuiInteractions.h"
#include "../GUI/ImGuiHelpers.h"
#include "TestAppHost.h"

namespace
{
struct InspectorFixture
{
    int values[2]{};
    int selected = 0;
    bool locked = false;
    bool longForm = false;
    float inspectorWidth = 320;
    float cardWidth = 200;
    float available = 0;
    float fieldWidth = 0;
    float explicitWidth = 0;
    float maxScroll = 0;
    int reveal = -1;
    int applied = -1;
    int changes = 0;
    ImGuiID ids[2]{};
    ImVec2 simpleCenter{}, lightCenter{}, lastCenter{}, cardSize{};

    void Draw()
    {
        ImGui::SetNextWindowPos({0, 0});
        ImGui::SetNextWindowSize({1100, 680});
        ImGui::Begin("Inspector fixture", nullptr, ImGuiWindowFlags_NoDecoration);
        for (int i = 0; i < 2; ++i)
        {
            if (i > 0) ImGui::SameLine();
            // Intentionally duplicate the saved/display name. IDs must still
            // distinguish two instances of the same processing tool.
            acmb::gui::RenderCompactStepCard(&values[i], "Stacker##same", {cardWidth, 110}, "Processing");
            cardSize = ImGui::GetItemRectSize();
        }
        ImGui::SetCursorPos({650, 8});
        ImGui::BeginChild("inspector", {inspectorWidth, 520}, ImGuiChildFlags_Borders,
            ImGuiWindowFlags_NoScrollbar);
        acmb::gui::RenderStepSettings(&values[selected], "Stacker##same", locked, [&](float width) {
            available = width;
            ids[selected] = ImGui::GetID("Simple");
            const auto center = [] {
                const auto min = ImGui::GetItemRectMin(), max = ImGui::GetItemRectMax();
                return ImVec2{(min.x + max.x) / 2, (min.y + max.y) / 2};
            };
            if (acmb::gui::RadioSetting("Light", values[selected], 0, false)) ++changes;
            lightCenter = center();
            if (acmb::gui::RadioSetting("Simple", values[selected], 1, false)) ++changes;
            simpleCenter = center();
            float number = 1;
            acmb::gui::FitSettingWidth("Width");
            ImGui::DragFloat("Width", &number);
            // ImGui's item rectangle includes its visible label.
            fieldWidth = ImGui::GetItemRectSize().x;
            ImGui::SetNextItemWidth(35);
            acmb::gui::FitSettingWidth("##column");
            ImGui::DragFloat("##column", &number);
            explicitWidth = ImGui::GetItemRectSize().x;
            if (longForm)
            {
                for (int i = 0; i < 30; ++i)
                {
                    const auto label = "Apply " + std::to_string(i);
                    acmb::gui::CommandButton(label.c_str(), {0, 0}, false, [&] { applied = i; });
                    if (i == 29) lastCenter = center();
                    if (i == reveal) ImGui::SetScrollHereY(0.5f);
                }
            }
            maxScroll = ImGui::GetScrollMaxY();
        });
        ImGui::EndChild();
        ImGui::End();
    }
};
}

ACMB_TESTS_NAMESPACE_BEGIN

BEGIN_SUITE(StepPresentation)

BEGIN_TEST(SharedProgressUsesOnlyFirstElement)
    struct Element { float progress; Element* next; float GetProgress() { return progress; } };
    Element tail{1, nullptr}, head{0.25f, &tail};
    EXPECT_EQ(0.25f, gui::PipelineHeadProgress(&head));
    tail.progress = 0;
    EXPECT_EQ(0.25f, gui::PipelineHeadProgress(&head));
    EXPECT_EQ(0.0f, gui::PipelineHeadProgress(static_cast<Element*>(nullptr)));
    head.progress = 1.2f;
    EXPECT_EQ(1.0f, gui::PipelineHeadProgress(&head));
    head.progress = NAN;
    EXPECT_EQ(0.0f, gui::PipelineHeadProgress(&head));
END_TEST

BEGIN_TEST(InspectorIdsAreDistinctAndStableAcrossSelection)
    TestAppHost host;
    InspectorFixture fixture;
    const auto draw = [&] { fixture.Draw(); };
    host.RenderFrame(draw);
    host.RenderFrame(draw);
    const auto firstId = fixture.ids[0];
    host.Click(fixture.simpleCenter, draw);
    EXPECT_EQ(1, fixture.values[0]);
    fixture.selected = 1;
    host.RenderFrame(draw);
    host.RenderFrame(draw);
    EXPECT_TRUE(firstId != fixture.ids[1]);
    host.Click(fixture.simpleCenter, draw);
    EXPECT_EQ(1, fixture.values[1]);
    host.Click(fixture.lightCenter, draw);
    EXPECT_EQ(0, fixture.values[1]);
    EXPECT_EQ(1, fixture.values[0]);
    fixture.selected = 0;
    host.RenderFrame(draw);
    EXPECT_EQ(firstId, fixture.ids[0]);
END_TEST

BEGIN_TEST(ControlWidthsFollowInspectorInsteadOfCards)
    TestAppHost host;
    InspectorFixture fixture;
    const auto draw = [&] { fixture.Draw(); };
    host.RenderFrame(draw);
    host.RenderFrame(draw);
    const auto original = fixture.available;
    EXPECT_TRUE(fixture.fieldWidth <= original + 1);
    EXPECT_EQ(35.0f, fixture.explicitWidth);
    fixture.cardWidth = 280;
    host.RenderFrame(draw);
    EXPECT_EQ(original, fixture.available);
    fixture.inspectorWidth = 400;
    host.RenderFrame(draw);
    host.RenderFrame(draw);
    EXPECT_TRUE(fixture.available > original);
    EXPECT_TRUE(fixture.fieldWidth <= fixture.available + 1);
END_TEST

BEGIN_TEST(InspectorDisablesExistingControlsDuringProcessing)
    TestAppHost host;
    InspectorFixture fixture;
    const auto draw = [&] { fixture.Draw(); };
    fixture.locked = true;
    host.RenderFrame(draw);
    host.RenderFrame(draw);
    host.Click(fixture.simpleCenter, draw);
    EXPECT_EQ(0, fixture.values[0]);
    EXPECT_EQ(0, fixture.changes);
    fixture.locked = false;
    host.Click(fixture.simpleCenter, draw);
    EXPECT_EQ(1, fixture.values[0]);
    EXPECT_EQ(1, fixture.changes);
END_TEST

BEGIN_TEST(LongSettingsScrollWithoutExpandingCards)
    TestAppHost host;
    InspectorFixture fixture;
    fixture.longForm = true;
    const auto draw = [&] { fixture.Draw(); };
    host.RenderFrame(draw);
    host.RenderFrame(draw);
    EXPECT_TRUE(fixture.maxScroll > 0);
    fixture.reveal = 29;
    host.RenderFrame(draw);
    host.RenderFrame(draw);
    host.Click(fixture.lastCenter, draw);
    EXPECT_EQ(29, fixture.applied);
    EXPECT_EQ(200.0f, fixture.cardSize.x);
    EXPECT_EQ(110.0f, fixture.cardSize.y);
END_TEST

BEGIN_TEST(PreviewHistogramSeparatesChannelsAndCountsEveryPixel)
    const unsigned char rows[2][8] = {
        {0, 128, 255, 255, 128, 0, 99, 99},
        {0, 64, 255, 127, 64, 127, 99, 99}
    };
    const auto bins = gui::BuildPreviewHistogram(2, 2, [&](uint32_t y) { return rows[y]; });
    EXPECT_EQ(2.0f, bins[0][0]);
    EXPECT_EQ(1.0f, bins[0][255]);
    EXPECT_EQ(1.0f, bins[0][127]);
    EXPECT_EQ(2.0f, bins[1][128]);
    EXPECT_EQ(2.0f, bins[1][64]);
    EXPECT_EQ(2.0f, bins[2][255]);
    EXPECT_EQ(1.0f, bins[2][0]);
    EXPECT_EQ(1.0f, bins[2][127]);
    for (const auto& channel : bins)
    {
        float total = 0;
        for (float count : channel) total += count;
        EXPECT_EQ(4.0f, total);
        EXPECT_EQ(0.0f, channel[99]);
    }
END_TEST

BEGIN_TEST(EmptyPreviewHistogramDoesNotReadPixels)
    bool read = false;
    const auto bins = gui::BuildPreviewHistogram(0, 0, [&](uint32_t) {
        read = true;
        return static_cast<const unsigned char*>(nullptr);
    });
    EXPECT_FALSE(read);
    for (const auto& channel : bins)
    {
        for (float count : channel)
        {
            EXPECT_EQ(0.0f, count);
        }
    }
END_TEST

BEGIN_TEST(PreviewHistogramUsesDisplayRgbaAndIgnoresAlpha)
    const unsigned char rgba[] = {0, 64, 255, 100, 255, 128, 0, 200};
    const auto bins = gui::BuildPreviewHistogram(2, 1, [&](uint32_t) { return rgba; }, 4);
    EXPECT_EQ(1.0f, bins[0][0]);
    EXPECT_EQ(1.0f, bins[0][255]);
    EXPECT_EQ(1.0f, bins[1][64]);
    EXPECT_EQ(1.0f, bins[1][128]);
    EXPECT_EQ(1.0f, bins[2][255]);
    EXPECT_EQ(1.0f, bins[2][0]);
    for (const auto& channel : bins)
    {
        EXPECT_EQ(0.0f, channel[100]);
        EXPECT_EQ(0.0f, channel[200]);
    }
END_TEST

BEGIN_TEST(ComboListDoesNotReadBeyondLastFormat)
    for (const auto& list : {std::string(".tif\0.y4m", 9), std::string(".tif\0.y4m\0", 10),
        std::string(".tif\0.y4m\0\0", 11)})
    {
        const auto items = UI::GetComboItems(list);
        EXPECT_EQ(size_t(2), items.size());
        EXPECT_EQ(std::string(".tif"), std::string(items[0]));
        EXPECT_EQ(std::string(".y4m"), std::string(items[1]));
    }
    EXPECT_TRUE(UI::GetComboItems("").empty());
END_TEST

END_SUITE

ACMB_TESTS_NAMESPACE_END
