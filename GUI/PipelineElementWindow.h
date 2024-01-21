#pragma once
#include "window.h"

#include "MenuItemsHolder.h"
#include "Texture.h"

#include "./../Core/bitmap.h"
#include "./../Geometry/point.h"

#if ( defined _MSC_VER && _MSC_VER > 1900 ) || ( __cplusplus > 202002L && __cpp_concepts >= 202002L )
#include <expected>
template <class T, class U>
using Expected = std::expected<T,U>;
using std::unexpected;
#else
#include "tl/expected.hpp"
template <class T, class U>
using Expected = tl::expected<T,U>;
using namespace tl;
#endif

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
    std::string _previewPopupName;

public:    

    static constexpr int cElementWidth = 150;
    static constexpr int cElementHeight = 250;

    enum class RelationType
    {
        None,
        Batch,
        Join
    };

    struct Relation
    {
        std::weak_ptr<PipelineElementWindow> pElement;
        RelationType relationType = RelationType::Batch;
    };

    struct SerializedInputs
    {
        RelationType left = RelationType::None;
        RelationType top = RelationType::None;
    };

    enum class PropagationDir
    {
        Forward = 1,
        Backward = 2,
        Both = 3
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

    int _inOutFlags{};
    SerializedInputs _serializedInputs;

    Point _gridPos;

    bool _primaryInputIsOnTop = false;
    IBitmapPtr _pSecondaryInputResult;

    int _remainingBytes{};

    bool _showError = false;
    std::string _error;

    std::unique_ptr<Texture> _pPreviewTexture;
    IBitmapPtr _pPreviewBitmap;
    bool _showPreview = false;

    PipelineElementWindow( const std::string& name, const Point& gridPos, int inOutFlags );

    virtual void DrawPipelineElementControls() = 0;
    
    virtual Expected<IBitmapPtr, std::string> RunTask( size_t i );
    virtual IBitmapPtr ProcessBitmapFromPrimaryInput( IBitmapPtr pSource, size_t taskNumber = 0 ) = 0;
 
    Expected<IBitmapPtr, std::string> ProcessSecondaryInput();
    virtual ImGuiWindowFlags flags() const override { return ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoDecoration; }
    virtual bool DrawHeader() override;

public:

    Expected<IBitmapPtr, std::string> RunTaskAndReportProgress( size_t i );

    std::shared_ptr<PipelineElementWindow> GetPrimaryInput() const;
    std::shared_ptr<PipelineElementWindow> GetSecondaryInput() const;

    std::shared_ptr<PipelineElementWindow> GetLeftInput() const;
    std::shared_ptr<PipelineElementWindow> GetTopInput() const;
    void SetLeftInput( std::shared_ptr<PipelineElementWindow> pPrimaryInput );
    void SetTopInput( std::shared_ptr<PipelineElementWindow> pSecondaryInput );

    std::shared_ptr<PipelineElementWindow> GetRightOutput() const;
    std::shared_ptr<PipelineElementWindow> GetBottomOutput() const;

    void SetRightOutput( std::shared_ptr<PipelineElementWindow> pElement );
    void SetBottomOutput( std::shared_ptr<PipelineElementWindow> pElement );

    RelationType GetLeftRelationType() const { return _leftInput.relationType; }
    RelationType GetTopRelationType() const { return _topInput.relationType; }
    RelationType GetRightRelationType() const { return _rightOutput.relationType; }
    RelationType GetBottomRelationType() const { return _bottomOutput.relationType; }

    void SetLeftRelationType( RelationType val ) { _leftInput.relationType = val; _serializedInputs.left = val; }
    void SetTopRelationType( RelationType val ) { _topInput.relationType = val; _serializedInputs.top = val; }
    void SetRightRelationType( RelationType val ) { _rightOutput.relationType = val; }
    void SetBottomRelationType( RelationType val ) { _bottomOutput.relationType = val; }

    int GetInOutFlags() const;

    bool HasFreeInputs();
    bool HasFreeOutputs();

    virtual size_t GetTaskCount();
    size_t GetCompletedTaskCount();
    virtual void ResetTasks();
    void ResetProgress( PropagationDir dir );

    virtual uint8_t GetMenuOrder() const = 0;

    virtual void Serialize(std::ostream& out) const;
    virtual void Deserialize(std::istream& in);
    virtual int GetSerializedStringSize() const;

    SerializedInputs GetActualInputs()  const { return _serializedInputs; }

    virtual std::string GetTaskName( size_t taskNumber ) const
    {
        auto pPrimaryInput = GetPrimaryInput();
        return pPrimaryInput ? pPrimaryInput->GetTaskName( taskNumber ) : std::string{};
    }

    IBitmapPtr GetPreviewBitmap() const { return _pPreviewBitmap; }
    Expected<void, std::string> GeneratePreviewTexture();
    
    void ResetPreview();

protected:
    virtual void DrawDialog() override;
    virtual Expected<void, std::string> GeneratePreviewBitmap() = 0;
    virtual Expected<Size, std::string> GetBitmapSize();
};

#define SET_MENU_PARAMS( ICON, CAPTION, TOOLTIP, ORDER ) \
inline static const std::string icon = ICON;\
inline static const std::string caption = CAPTION;\
inline static const std::string tooltip = TOOLTIP;\
inline static const int order = ORDER;\
virtual uint8_t GetMenuOrder() const override { return order; }

ACMB_GUI_NAMESPACE_END
