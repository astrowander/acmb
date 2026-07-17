#pragma once
#include "window.h"

#include "MenuItemsHolder.h"
#include "Texture.h"
#include "AsyncWorker.h"

#include "./../Core/bitmap.h"
#include "./../Geometry/point.h"

#if ( defined _MSC_VER && _MSC_VER > 1900 ) || ( __cplusplus > 202002L && __cpp_concepts >= 202002L )
#include <expected>
template <class T, class U>
using Expected = std::expected<T,U>;
using std::unexpected;
#else
#include "expected.hpp"
template <class T, class U>
using Expected = tl::expected<T,U>;
using namespace tl;
#endif

ACMB_GUI_NAMESPACE_BEGIN

enum PEFlags : int
{
    PEFlags_NoOutput = 1,
    PEFlags_StrictlyOneOutput = 2,    
    PEFlags_NoInput = 8,
    PEFlags_StrictlyOneInput = 16
};

class PipelineElementWindow : public Window
{
public:    

    static constexpr int cElementWidth = 150;
    static constexpr int cElementHeight = 250;

    enum class PropagationDir
    {
        None = 0,
        Forward = 1,
        Backward = 2,
        Both = 3
    };

protected:

    float _itemWidth = 0.0f;

    size_t _taskCount = 0;
    size_t _completedTaskCount = 0;
    float _taskReadiness = 0.0f;

    int _remainingBytes{};

    bool _showError = false;
    std::string _error;

    bool _needToUpdatePreview = false;
    std::shared_ptr<Texture> _pPreviewTexture;
    std::atomic<IBitmapPtr> _pPreviewBitmap;

    std::list<AsyncWorker> _previewWorkers;

    std::shared_ptr<PipelineElementWindow> _output;
    std::weak_ptr<PipelineElementWindow> _input;

    PipelineElementWindow( const std::string& name );

    virtual void DrawPipelineElementControls() = 0;
    
    virtual Expected<IBitmapPtr, std::string> RunTask( size_t i );
    virtual IBitmapPtr ProcessBitmapFromPrimaryInput( IBitmapPtr pSource, size_t taskNumber = 0 ) = 0;
    virtual ImGuiWindowFlags flags() const override { return ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoDecoration; }
    virtual bool DrawHeader() override;

public:

    Expected<IBitmapPtr, std::string> RunTaskAndReportProgress( size_t i );

    Expected<IBitmapPtr, std::string> GetInputPreview(bool forNextElement, bool fullSize) const;
    std::shared_ptr<PipelineElementWindow> GetInput() const;
    void SetInput( std::shared_ptr<PipelineElementWindow> pPrimaryInput );

    std::shared_ptr<PipelineElementWindow> GetOutput() const;
    void SetOutput( std::shared_ptr<PipelineElementWindow> pElement );

    int GetFollowingElementsCount() const;


    virtual size_t GetTaskCount(bool update = false);
    size_t GetCompletedTaskCount();
    virtual void ResetTasks();
    void ResetProgress( PropagationDir dir );

    virtual uint8_t GetMenuOrder() const = 0;

    virtual void Serialize(std::ostream& out) const;

    virtual bool Deserialize(std::istream& in);
    virtual int GetSerializedStringSize() const;

    virtual std::string GetTaskName( size_t taskNumber ) const;

    std::string GeneratePreviewBitmapAsync( bool forNextElement, bool fullSize );
    Expected<std::shared_ptr<Texture>, std::string> GetPreviewTexture();
    
    void ResetPreview(PropagationDir dir);

    virtual Expected<Size, std::string> GetBitmapSize();

    virtual void OnInputChanged() {}

    virtual void OnPreviewedFrameNumberChanged( int val );

    virtual void OnKeyframeCommited() 
    {
    }

    virtual int GetPreviewedFrameNumber() const 
    { 
        if ( auto pInput = GetInput() ) 
            return pInput->GetPreviewedFrameNumber(); 
        else 
            return -1; 
    }

    virtual void SetPreviewedFrameNumber( int val ) 
    { 
        if ( auto pInput = GetInput() ) 
            pInput->SetPreviewedFrameNumber( val ); 
    }

    int GetElementsCount() const;

    virtual void DrawOnPreviewImage( ImDrawList* pDrawList, ImVec2 topLeftPos, ImVec2 previewSize ) {}

    virtual Expected<IBitmapPtr, std::string> GeneratePreviewBitmap(bool forNextElement, bool fullSize) = 0;

protected:
    virtual void DrawDialog() override;
};

#define SET_MENU_PARAMS( ICON, CAPTION, TOOLTIP, ORDER ) \
inline static const std::string icon = ICON;\
inline static const std::string caption = CAPTION;\
inline static const std::string tooltip = TOOLTIP;\
inline static const int order = ORDER;\
virtual uint8_t GetMenuOrder() const override { return order; }

ACMB_GUI_NAMESPACE_END
