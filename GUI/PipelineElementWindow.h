#pragma once
#include "window.h"

#include "./../Core/IPipelineElement.h"
#include "./../Core/bitmap.h"
#include "./../Geometry/point.h"

#include <expected>

ACMB_GUI_NAMESPACE_BEGIN

class PipelineElementWindow : public Window
{
public:
    enum RequiredInOutFlags
    {
        NoOutput = 1,
        StrictlyOneOutput = 2,
        // gap for possible StrictlyTwoOutputs
        NoInput = 8,
        StrictlyOneInput = 16,
        StrictlyTwoInputs = 32
    };

    static constexpr int cElementWidth = 150;
    static constexpr int cElementHeight = 250;

protected:

    float _itemWidth = 0.0f;

    size_t _taskCount = 0;
    size_t _completedTaskCount = 0;

    std::weak_ptr<PipelineElementWindow> _pLeftInput;
    std::weak_ptr<PipelineElementWindow> _pBottomInput;

    int _inOutFlags;

    Point _gridPos;

    PipelineElementWindow( const std::string& name, const Point& gridPos, int inOutFlags )
    : Window( name + "##R" + std::to_string(gridPos.y) + "C" + std::to_string(gridPos.x), {cElementWidth, cElementHeight})
    , _inOutFlags( inOutFlags )
    , _itemWidth( cElementWidth - ImGui::GetStyle().WindowPadding.x * cMenuScaling )
    {
    }

    virtual void DrawPipelineElementControls() = 0;
    virtual std::expected<IBitmapPtr, std::string> RunTask( size_t i ) = 0;

    virtual ImGuiWindowFlags flags() override { return ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoFocusOnAppearing; }

public:

    std::expected<IBitmapPtr, std::string> RunTaskAndReportProgress( size_t i );

    auto GetLeftInput()
    {
        return _pLeftInput.lock();
    }
    auto GetBottomInput()
    {
        return _pBottomInput.lock();
    }
    void SetLeftInput( std::shared_ptr<PipelineElementWindow> pPrimaryInput )
    {
        _pLeftInput = pPrimaryInput;
    }
    void SetBottomInput( std::shared_ptr<PipelineElementWindow> pSecondaryInput )
    {
        _pBottomInput = pSecondaryInput;
    }

    size_t GetTaskCount()
    {
        if ( _taskCount == 0 )
        {
            auto pPrimaryInput = GetLeftInput();
            if ( pPrimaryInput )
                _taskCount = pPrimaryInput->GetTaskCount();
        }

        return _taskCount;
    }
    auto GetInOutFlags()
    {
        return _inOutFlags;
    }

protected:

    virtual void DrawDialog() override;
};



ACMB_GUI_NAMESPACE_END
