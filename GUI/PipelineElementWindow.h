#pragma once
#include "window.h"

#include "./../Core/IPipelineElement.h"
#include "./../Core/bitmap.h"

#include <expected>
#include <functional>

ACMB_GUI_NAMESPACE_BEGIN

class PipelineElementWindow : public Window
{
    IPipelineElementPtr _pElement;
    

protected:
    size_t _taskCount = 0;
    size_t _completedTaskCount = 0;

    std::weak_ptr<PipelineElementWindow> _pPrimaryInput;
    std::weak_ptr<PipelineElementWindow> _pSecondaryInput;

    PipelineElementWindow( IPipelineElementPtr pElement, const std::string& name, const ImVec2& pos, const ImVec2& size, std::shared_ptr<Window> pParent )
        : Window( name, pos, size, pParent )
        , _pElement( pElement )
    {
    }

    virtual void DrawPipelineElementControls() = 0;
    virtual std::expected<IBitmapPtr, std::string> RunTask( size_t i ) = 0;

public:

    std::expected<IBitmapPtr, std::string> RunTaskAndReportProgress( size_t i );

    auto GetPrimaryInput()
    {
        return _pPrimaryInput.lock();
    }
    auto GetSecondaryInput()
    {
        return _pSecondaryInput.lock();
    }
    void SetPrimaryInput( std::shared_ptr<PipelineElementWindow> pPrimaryInput )
    {
        _pPrimaryInput = pPrimaryInput;
    }
    void SetSecondaryInput( std::shared_ptr<PipelineElementWindow> pSecondaryInput )
    {
        _pSecondaryInput = pSecondaryInput;
    }

    auto GetTaskCount()
    {
        return _taskCount;
    }
    auto GetElement()
    {
        return _pElement;
    }

protected:

    virtual void DrawDialog() override;
};



ACMB_GUI_NAMESPACE_END
