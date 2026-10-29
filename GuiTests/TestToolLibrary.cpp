#include "../Tests/test.h"
#include "../GUI/ToolLibrary.h"
#include "TestAppHost.h"

namespace
{
constexpr int ToolIds[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16};

struct LibraryFixture
{
    acmb::gui::Items items;
    std::vector<int> calledIds;
    size_t lastPosition = 0;
    bool lastReplace = false;
    bool locked = false;
    std::map<int, ImVec2> tools;
    std::map<std::string, ImVec2> headings;

    LibraryFixture()
    {
        for (int id : ToolIds)
            items["Tools"].emplace(id, std::make_unique<acmb::gui::MenuItem>("+", "Tool " + std::to_string(id), "",
                [this, id](size_t position, bool replace) {
                    calledIds.push_back(id); lastPosition = position; lastReplace = replace;
                }, false, acmb::gui::MenuItemsHolder::GetToolPresentation(uint8_t(id))));
    }

    void Draw()
    {
        tools.clear();
        headings.clear();
        ImGui::SetNextWindowPos({0, 0});
        ImGui::SetNextWindowSize({500, 650});
        ImGui::Begin("Library fixture", nullptr, ImGuiWindowFlags_NoDecoration);
        acmb::gui::BeginVerticalPanel("library", {340, 550}, 220);
        acmb::gui::DrawToolLibrary(acmb::gui::MenuItemsHolder::BuildToolGroups(items),
            [this] { return locked; }, 3, true, {}, [this](const char* group, int id) {
                const auto min = ImGui::GetItemRectMin();
                const auto max = ImGui::GetItemRectMax();
                const ImVec2 center{(min.x + max.x) / 2, (min.y + max.y) / 2};
                if (id < 0) headings[group] = center;
                else tools[id] = center;
            });
        ImGui::EndChild();
        ImGui::End();
    }
};
}

ACMB_TESTS_NAMESPACE_BEGIN

BEGIN_SUITE(ToolLibrary)

BEGIN_TEST(GroupsMatchRequestedDisplayOrder)
    LibraryFixture fixture;
    const auto groups = gui::MenuItemsHolder::BuildToolGroups(fixture.items);
    EXPECT_EQ(size_t(5), groups.size());
    const char* captions[] = {"Input / Output", "Stack & Calibration", "Geometry", "Adjustments", "Cleanup"};
    const std::vector<std::vector<int>> expected{{1, 2, 8}, {3, 4, 5}, {6, 7, 11, 14}, {9, 10}, {12, 15, 16}};
    for (size_t group = 0; group < expected.size(); ++group)
    {
        EXPECT_EQ(std::string(captions[group]), std::string(groups.at(group).caption));
        EXPECT_EQ(expected[group].size(), groups.at(group).tools.size());
        for (size_t item = 0; item < expected[group].size(); ++item)
        {
            const auto& entry = groups.at(group).tools.at(item);
            EXPECT_EQ(expected[group][item], entry.first);
            EXPECT_TRUE(entry.second == fixture.items.at("Tools").at(entry.first).get());
        }
    }
END_TEST

BEGIN_TEST(PresentationChangesPreserveProjectIdLookupAndCallbacks)
    LibraryFixture fixture;
    const auto* converter = fixture.items.at("Tools").at(8).get();
    fixture.items.at("Tools").at(8)->presentation = {gui::ToolGroup::Cleanup, 99};
    const auto groups = gui::MenuItemsHolder::BuildToolGroups(fixture.items);
    EXPECT_EQ(8, groups.back().tools.back().first);
    EXPECT_TRUE(fixture.items.at("Tools").at(8).get() == converter);
    EXPECT_EQ(size_t(15), fixture.items.at("Tools").size());

    // The existing project loader looks up byte IDs in the "Tools" registry.
    const std::string serializedIds{char(1), char(8), char(3), char(14), char(16), char(2)};
    for (size_t i = 0; i < serializedIds.size(); ++i)
    {
        const uint8_t id = static_cast<uint8_t>(serializedIds[i]);
        fixture.items.at("Tools").at(id)->action(i, true);
        EXPECT_EQ(int(id), fixture.calledIds.back());
        EXPECT_EQ(i, fixture.lastPosition);
        EXPECT_TRUE(fixture.lastReplace);
    }
END_TEST

BEGIN_TEST(UnknownToolsRemainVisibleWithoutChangingTheirId)
    LibraryFixture fixture;
    fixture.items["Tools"].emplace(13, std::make_unique<gui::MenuItem>("", "Additional tool", "",
        [](size_t, bool) {}, false, gui::MenuItemsHolder::GetToolPresentation(13)));
    const auto groups = gui::MenuItemsHolder::BuildToolGroups(fixture.items);
    EXPECT_EQ(std::string("Other"), std::string(groups.back().caption));
    EXPECT_EQ(13, groups.back().tools.front().first);
    EXPECT_TRUE(groups.back().tools.front().second == fixture.items.at("Tools").at(13).get());
    gui::Items empty;
    EXPECT_TRUE(gui::MenuItemsHolder::BuildToolGroups(empty).empty());
END_TEST

BEGIN_TEST(CollapsibleGroupsRetainActionsAndRespectLocks)
    TestAppHost host;
    LibraryFixture fixture;
    const auto draw = [&] { fixture.Draw(); };
    host.RenderFrame(draw);
    host.RenderFrame(draw);
    host.Click(fixture.tools.at(8), draw);
    EXPECT_EQ(size_t(1), fixture.calledIds.size());
    EXPECT_EQ(8, fixture.calledIds.back());
    EXPECT_EQ(size_t(3), fixture.lastPosition);
    EXPECT_TRUE(fixture.lastReplace);
    fixture.locked = true;
    host.Click(fixture.tools.at(1), draw);
    EXPECT_EQ(size_t(1), fixture.calledIds.size());
    host.Click(fixture.headings.at("Input / Output"), draw);
    EXPECT_FALSE(fixture.tools.contains(1));
    EXPECT_FALSE(fixture.tools.contains(2));
    EXPECT_FALSE(fixture.tools.contains(8));
    EXPECT_TRUE(fixture.tools.contains(3));
    host.Click(fixture.headings.at("Input / Output"), draw);
    EXPECT_TRUE(fixture.tools.contains(8));
    fixture.locked = false;
    host.Click(fixture.tools.at(1), draw);
    EXPECT_EQ(size_t(2), fixture.calledIds.size());
    EXPECT_EQ(1, fixture.calledIds.back());
END_TEST

END_SUITE

ACMB_TESTS_NAMESPACE_END
