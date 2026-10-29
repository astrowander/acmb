#include "../Tests/test.h"
#include "../GUI/MenuItem.h"
#include "imgui.h"
#include "TestAppHost.h"

ACMB_TESTS_NAMESPACE_BEGIN

BEGIN_SUITE(GuiSmoke)

BEGIN_TEST(RendersHeadlessFrames)
    TestAppHost host;
    for (int frame = 0; frame < 2; ++frame)
    {
        host.RenderFrame([] {
        ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
        ImGui::SetNextWindowSize(ImVec2(320.0f, 240.0f));
        ImGui::Begin("GUI smoke test");
        ImGui::Button("Run###RunCommand");
        ImGui::End();
        });

        const auto* drawData = ImGui::GetDrawData();
        EXPECT_TRUE(drawData != nullptr);
        EXPECT_TRUE(drawData->Valid);
        EXPECT_TRUE(drawData->TotalVtxCount > 0);
    }
END_TEST

BEGIN_TEST(MenuActionPreservesSelectionArguments)
    size_t selectedPosition = 0;
    bool replaceSelected = false;
    gui::MenuItem item("", "Stack", "Add a stacker",
        [&](size_t position, bool replace)
        {
            selectedPosition = position;
            replaceSelected = replace;
        });

    item.action(3, true);
    EXPECT_EQ(size_t(3), selectedPosition);
    EXPECT_TRUE(replaceSelected);
END_TEST

END_SUITE

ACMB_TESTS_NAMESPACE_END
