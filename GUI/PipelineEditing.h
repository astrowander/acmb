#pragma once

#include "../Core/macros.h"
#include <algorithm>
#include <memory>
#include <stdexcept>

ACMB_GUI_NAMESPACE_BEGIN

// Selection alternates between a step and the insertion position before it.
// Index == pipelineSize is the empty position after the final step.
struct PipelineSelection
{
    int activeIndex = 0;
    int firstVisible = 0;
    bool elementSelected = true;

    void Select(int index, bool selected, int pipelineSize, int visibleCells)
    {
        if (index < 0)
        {
            index = 0;
            selected = false;
        }
        if (index > pipelineSize)
        {
            index = pipelineSize;
            selected = true;
        }
        activeIndex = index;
        elementSelected = selected;
        firstVisible = std::min(firstVisible, activeIndex);
        const int count = std::max(1, visibleCells);
        if (activeIndex >= firstVisible + count)
            firstVisible = activeIndex - count + 1;
    }

    void MoveLeft(int pipelineSize, int visibleCells)
    {
        Select(elementSelected ? activeIndex : activeIndex - 1,
            !elementSelected, pipelineSize, visibleCells);
    }

    void MoveRight(int pipelineSize, int visibleCells)
    {
        Select(elementSelected ? activeIndex + 1 : activeIndex,
            !elementSelected, pipelineSize, visibleCells);
    }
};

template<class Node>
std::shared_ptr<Node> GetPipelineElement(std::shared_ptr<Node> head, size_t position)
{
    while (head && position > 0)
    {
        head = head->GetOutput();
        --position;
    }
    return head;
}

// The new node must be fresh and unlinked. Calls the actual node setters so
// production OnInputChanged notifications continue to run.
template<class Node>
void InsertPipelineElement(std::shared_ptr<Node>& head, std::shared_ptr<Node> node,
    size_t position, bool replace)
{
    if (!node)
        throw std::invalid_argument("Cannot insert a null pipeline element");
    auto current = head;
    std::shared_ptr<Node> previous;
    for (size_t i = 0; i < position; ++i)
    {
        if (!current)
            throw std::out_of_range("Pipeline insertion position is past the end");
        previous = current;
        current = current->GetOutput();
    }
    const auto next = replace && current ? current->GetOutput() : current;
    if (previous)
        node->SetInput(previous);
    if (previous)
        previous->SetOutput(node);
    else
        head = node;
    node->SetOutput(next);
    if (next)
        next->SetInput(node);
}

template<class Node>
bool DeletePipelineElement(std::shared_ptr<Node>& head, size_t position)
{
    auto current = GetPipelineElement(head, position);
    if (!current)
        return false;
    auto previous = current->GetInput();
    auto next = current->GetOutput();
    if (previous)
        previous->SetOutput(next);
    else
        head = next;
    if (next)
        next->SetInput(previous);
    return true;
}

template<class Node>
void ClearEditedPipeline(std::shared_ptr<Node>& head, PipelineSelection& selection)
{
    head.reset();
    selection = {};
}

ACMB_GUI_NAMESPACE_END
