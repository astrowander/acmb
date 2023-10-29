#pragma once
#include "window.h"

#include "MenuItemsHolder.h"

#include "./../Core/IPipelineElement.h"
#include "./../Core/bitmap.h"
#include "./../Geometry/point.h"

#include <expected>

ACMB_GUI_NAMESPACE_BEGIN


enum PEFlags : int
{
    PEFlags_NoOutput = 1,
    PEFlags_StrictlyOneOutput = 2,
    // gap for possible StrictlyTwoOutputs
    PEFlags_NoInput = 8,
    PEFlags_StrictlyOneInput = 16,
    PEFlags_StrictlyTwoInputs = 32
};

class PipelineElementWindow : public Window
{
    bool _openRenamePopup = false;
    std::array<char, 256> _renameBuf = {};

public:    

    static constexpr int cElementWidth = 150;
    static constexpr int cElementHeight = 250;

    enum class RelationType
    {
        Batch,
        Join
    };

    struct Relation
    {
        std::weak_ptr<PipelineElementWindow> pElement;
        RelationType relationType = RelationType::Batch;
    };

protected:

    float _itemWidth = 0.0f;

    size_t _taskCount = 0;
    size_t _completedTaskCount = 0;
    float _taskReadiness = 0.0f;

    Relation _leftInput;
    Relation _topInput;

    Relation _rightOutput;
    Relation _bottomOutput;

    int _inOutFlags = {};
    char _actualInputs = {};

    Point _gridPos;

    PipelineElementWindow( const std::string& name, const Point& gridPos, int inOutFlags );

    virtual void DrawPipelineElementControls() = 0;
    virtual std::expected<IBitmapPtr, std::string> RunTask( size_t i ) = 0;

    virtual ImGuiWindowFlags flags() override { return ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoFocusOnAppearing; }

public:

    std::expected<IBitmapPtr, std::string> RunTaskAndReportProgress( size_t i );

    std::shared_ptr<PipelineElementWindow> GetLeftInput();
    std::shared_ptr<PipelineElementWindow> GetTopInput();
    void SetLeftInput( std::shared_ptr<PipelineElementWindow> pPrimaryInput );
    void SetTopInput( std::shared_ptr<PipelineElementWindow> pSecondaryInput );

    std::shared_ptr<PipelineElementWindow> GetRightOutput();
    std::shared_ptr<PipelineElementWindow> GetBottomOutput();

    void SetRightOutput( std::shared_ptr<PipelineElementWindow> pElement );
    void SetBottomOutput( std::shared_ptr<PipelineElementWindow> pElement );


    RelationType GetLeftRelationType() { return _leftInput.relationType; }
    RelationType GetTopRelationType() { return _topInput.relationType; }
    RelationType GetRightRelationType() { return _rightOutput.relationType; }
    RelationType GetBottomRelationType() { return _bottomOutput.relationType; }

    void SetLeftRelationType( RelationType val ) { _leftInput.relationType = val; }
    void SetTopRelationType( RelationType val )  { _topInput.relationType = val; }
    void SetRightRelationType( RelationType val ) { _rightOutput.relationType = val; }
    void SetBottomRelationType( RelationType val ) { _bottomOutput.relationType = val; }

    int GetInOutFlags();

    bool HasFreeInputs();
    bool HasFreeOutputs();

    size_t GetTaskCount();
    virtual void ResetTasks();

    virtual uint8_t GetMenuOrder() = 0;

    virtual void Serialize(std::ostream& out);
    virtual void Deserialize(std::istream& in);

    char GetActualInputs() { return _actualInputs; }

protected:
    virtual void DrawDialog() override;
};



ACMB_GUI_NAMESPACE_END
