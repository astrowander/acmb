#include "../Tests/test.h"
#include "../GUI/StepPresentation.h"
#include "../GUI/GuiInteractions.h"
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

END_SUITE

ACMB_TESTS_NAMESPACE_END
