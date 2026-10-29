#include "../Tests/test.h"
#include "../GUI/PipelineEditing.h"
#include <limits>

namespace
{
// Only the link API is substituted. Editing and selection run the same
// production algorithms as MainWindow, without constructing processing tools.
struct EditingNode
{
    int id;
    int inputChanges = 0;
    std::weak_ptr<EditingNode> input;
    std::shared_ptr<EditingNode> output;

    explicit EditingNode(int value) : id(value) {}
    auto GetInput() const { return input.lock(); }
    auto GetOutput() const { return output; }
    void SetInput(std::shared_ptr<EditingNode> node) { input = node; ++inputChanges; }
    void SetOutput(std::shared_ptr<EditingNode> node) { output = node; }
};

using NodePtr = std::shared_ptr<EditingNode>;

NodePtr MakePipeline(int count = 3)
{
    NodePtr head;
    for (int i = 0; i < count; ++i)
        acmb::gui::InsertPipelineElement(head, std::make_shared<EditingNode>(i), size_t(i), false);
    return head;
}

bool Matches(NodePtr head, const std::vector<int>& expected)
{
    NodePtr previous;
    for (int id : expected)
    {
        if (!head || head->id != id || head->GetInput() != previous)
            return false;
        previous = head;
        head = head->GetOutput();
    }
    if (head)
        return false;
    // Verify reverse traversal as well, with bounded iteration to catch cycles.
    for (auto it = expected.rbegin(); it != expected.rend(); ++it)
    {
        if (!previous || previous->id != *it)
            return false;
        previous = previous->GetInput();
    }
    return !previous;
}
}

ACMB_TESTS_NAMESPACE_BEGIN

BEGIN_SUITE(PipelineEditing)

BEGIN_TEST(InsertIntoEmptyPipeline)
    NodePtr head;
    gui::InsertPipelineElement(head, std::make_shared<EditingNode>(7), 0, false);
    EXPECT_TRUE(Matches(head, {7}));
END_TEST

BEGIN_TEST(InsertAtHeadMiddleAndTail)
    for (size_t position = 0; position <= 3; ++position)
    {
        auto head = MakePipeline();
        auto inserted = std::make_shared<EditingNode>(7);
        gui::InsertPipelineElement(head, inserted, position, false);
        std::vector<int> expected{0, 1, 2};
        expected.insert(expected.begin() + position, 7);
        EXPECT_TRUE(Matches(head, expected));
        EXPECT_TRUE(gui::GetPipelineElement(head, position) == inserted);
    }
END_TEST

BEGIN_TEST(ReplaceAtHeadMiddleAndTail)
    for (size_t position = 0; position < 3; ++position)
    {
        auto head = MakePipeline();
        std::weak_ptr<EditingNode> removed = gui::GetPipelineElement(head, position);
        gui::InsertPipelineElement(head, std::make_shared<EditingNode>(7), position, true);
        std::vector<int> expected{0, 1, 2};
        expected[position] = 7;
        EXPECT_TRUE(Matches(head, expected));
        EXPECT_TRUE(removed.expired());
    }
END_TEST

BEGIN_TEST(ReplaceAtEndAppendsIncludingEmptyPipeline)
    for (int size : {0, 3})
    {
        auto head = MakePipeline(size);
        gui::InsertPipelineElement(head, std::make_shared<EditingNode>(7), size_t(size), true);
        EXPECT_TRUE(Matches(head, size == 0 ? std::vector<int>{7} : std::vector<int>{0, 1, 2, 7}));
    }
END_TEST

BEGIN_TEST(InvalidInsertionLeavesPipelineUnchanged)
    auto head = MakePipeline();
    auto originalHead = head;
    for (bool replace : {false, true})
    {
        bool rejected = false;
        try { gui::InsertPipelineElement(head, std::make_shared<EditingNode>(7), 4, replace); }
        catch (const std::out_of_range&) { rejected = true; }
        EXPECT_TRUE(rejected);
        EXPECT_TRUE(head == originalHead);
        EXPECT_TRUE(Matches(head, {0, 1, 2}));
    }
    bool rejected = false;
    try { gui::InsertPipelineElement(head, NodePtr{}, 1, false); }
    catch (const std::invalid_argument&) { rejected = true; }
    EXPECT_TRUE(rejected);
    EXPECT_TRUE(Matches(head, {0, 1, 2}));
END_TEST

BEGIN_TEST(DeleteHeadMiddleAndTail)
    for (size_t position = 0; position < 3; ++position)
    {
        auto head = MakePipeline();
        std::weak_ptr<EditingNode> removed = gui::GetPipelineElement(head, position);
        EXPECT_TRUE(gui::DeletePipelineElement(head, position));
        std::vector<int> expected{0, 1, 2};
        expected.erase(expected.begin() + position);
        EXPECT_TRUE(Matches(head, expected));
        EXPECT_TRUE(removed.expired());
    }
END_TEST

BEGIN_TEST(DeleteOnlyElementAndInvalidPositions)
    auto head = MakePipeline(1);
    EXPECT_TRUE(gui::DeletePipelineElement(head, 0));
    EXPECT_TRUE(head == nullptr);
    EXPECT_FALSE(gui::DeletePipelineElement(head, 0));
    head = MakePipeline();
    EXPECT_FALSE(gui::DeletePipelineElement(head, 3));
    EXPECT_FALSE(gui::DeletePipelineElement(head, std::numeric_limits<size_t>::max()));
    EXPECT_TRUE(Matches(head, {0, 1, 2}));
END_TEST

BEGIN_TEST(EditsNotifySuccessorOfInputChanges)
    auto head = MakePipeline();
    auto successor = gui::GetPipelineElement(head, 1);
    int changes = successor->inputChanges;
    gui::InsertPipelineElement(head, std::make_shared<EditingNode>(7), 1, false);
    EXPECT_EQ(changes + 1, successor->inputChanges);
    EXPECT_TRUE(gui::DeletePipelineElement(head, 1));
    EXPECT_EQ(changes + 2, successor->inputChanges);
    gui::InsertPipelineElement(head, std::make_shared<EditingNode>(8), 0, true);
    EXPECT_EQ(changes + 3, successor->inputChanges);
    EXPECT_TRUE(Matches(head, {8, 1, 2}));
END_TEST

BEGIN_TEST(ClearReleasesNodesAndResetsSelection)
    auto head = MakePipeline();
    std::vector<std::weak_ptr<EditingNode>> nodes;
    for (auto node = head; node; node = node->GetOutput())
        nodes.push_back(node);
    gui::PipelineSelection selection;
    selection.Select(3, false, 3, 1);
    gui::ClearEditedPipeline(head, selection);
    EXPECT_TRUE(head == nullptr);
    EXPECT_EQ(0, selection.activeIndex);
    EXPECT_EQ(0, selection.firstVisible);
    EXPECT_TRUE(selection.elementSelected);
    for (const auto& node : nodes)
        EXPECT_TRUE(node.expired());
    gui::ClearEditedPipeline(head, selection);
    gui::InsertPipelineElement(head, std::make_shared<EditingNode>(7), 0, false);
    EXPECT_TRUE(Matches(head, {7}));
END_TEST

BEGIN_TEST(KeyboardNavigationAlternatesStepsAndInsertionPositions)
    gui::PipelineSelection selection;
    selection.MoveLeft(3, 2);
    EXPECT_EQ(0, selection.activeIndex);
    EXPECT_FALSE(selection.elementSelected);
    selection.MoveLeft(3, 2);
    EXPECT_EQ(0, selection.activeIndex);
    EXPECT_FALSE(selection.elementSelected);
    for (int position = 0; position < 3; ++position)
    {
        selection.MoveRight(3, 2);
        EXPECT_EQ(position, selection.activeIndex);
        EXPECT_TRUE(selection.elementSelected);
        selection.MoveRight(3, 2);
        EXPECT_EQ(position + 1, selection.activeIndex);
        EXPECT_FALSE(selection.elementSelected);
    }
    selection.MoveRight(3, 2);
    selection.MoveRight(3, 2);
    EXPECT_EQ(3, selection.activeIndex);
    EXPECT_TRUE(selection.elementSelected);
    selection.MoveLeft(3, 2);
    EXPECT_EQ(3, selection.activeIndex);
    EXPECT_FALSE(selection.elementSelected);
    selection.MoveLeft(3, 2);
    EXPECT_EQ(2, selection.activeIndex);
    EXPECT_TRUE(selection.elementSelected);
END_TEST

BEGIN_TEST(SelectionClampsToPipelineAndKeepsActivePositionVisible)
    gui::PipelineSelection selection;
    selection.Select(8, true, 10, 3);
    EXPECT_EQ(6, selection.firstVisible);
    selection.Select(1, false, 10, 3);
    EXPECT_EQ(1, selection.firstVisible);
    selection.Select(100, false, 3, 2);
    EXPECT_EQ(3, selection.activeIndex);
    EXPECT_TRUE(selection.elementSelected);
    EXPECT_EQ(2, selection.firstVisible);
    selection.Select(-1, true, 3, 2);
    EXPECT_EQ(0, selection.activeIndex);
    EXPECT_FALSE(selection.elementSelected);
    EXPECT_EQ(0, selection.firstVisible);
END_TEST

BEGIN_TEST(SelectionSurvivesDeletionAndViewportShrink)
    auto head = MakePipeline();
    gui::PipelineSelection selection;
    selection.Select(2, true, 3, 3);
    selection.Select(selection.activeIndex, selection.elementSelected, 3, 1);
    EXPECT_EQ(2, selection.firstVisible);
    EXPECT_TRUE(gui::DeletePipelineElement(head, 2));
    selection.Select(selection.activeIndex, selection.elementSelected, 2, 1);
    EXPECT_EQ(2, selection.activeIndex); // The empty position after the new tail.
    EXPECT_TRUE(gui::GetPipelineElement(head, size_t(selection.activeIndex)) == nullptr);
    EXPECT_TRUE(gui::DeletePipelineElement(head, 1));
    selection.Select(selection.activeIndex, selection.elementSelected, 1, 0);
    EXPECT_EQ(1, selection.activeIndex);
    EXPECT_EQ(1, selection.firstVisible);
    EXPECT_TRUE(gui::DeletePipelineElement(head, 0));
    selection.Select(selection.activeIndex, selection.elementSelected, 0, 1);
    EXPECT_EQ(0, selection.activeIndex);
    EXPECT_EQ(0, selection.firstVisible);
END_TEST

END_SUITE

ACMB_TESTS_NAMESPACE_END
