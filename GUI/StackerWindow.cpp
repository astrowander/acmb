#include "StackerWindow.h"
#include "MainWindow.h"
#include "Serializer.h"
#include "ImGuiHelpers.h"
#include "./../Registrator/stacker.h"
#include "./../Cuda/CudaStacker.h"

ACMB_GUI_NAMESPACE_BEGIN

StackerWindow::StackerWindow( const Point& gridPos )
    : PipelineElementWindow( "Stacker", gridPos, PEFlags_StrictlyOneInput | PEFlags_StrictlyOneOutput )
{
    _taskCount = 1;
}

void StackerWindow::DrawPipelineElementControls()
{
    ImGui::Text( "Stack Mode" );
    UI::RadioButton( "Light Frames", ( int* ) (&_stackMode), int( StackMode::Light ), "Input images will be debayered and aligned by stars before stacking" );
    UI::RadioButton( "Light (w/o alignment)", ( int* ) (&_stackMode), int( StackMode::LightNoAlign ), "Input images will be stacked without alignment and then debayered" );
    UI::RadioButton( "Dark/Flat Frames", ( int* ) (&_stackMode), int( StackMode::DarkOrFlat ), "Input images will be stacked as-is, without alignment and debayerization" );

    ImGui::Separator();    
    if ( _stackMode == StackMode::Light )
    {
        ImGui::Text( "Star Detection Threshold" );
        UI::DragFloat( "##StarDetectionThreshold", &_threshold, 0.1f, 0.0f, 100.0f,
                       "A group of pixels will be recognized as a star only if their luminosity is greater than this threshold (in percents) above the median value" );
    }
}

Expected<IBitmapPtr, std::string> StackerWindow::RunTask( size_t )
{
    _completedTaskCount = 0;

    auto pInput = GetPrimaryInput();
    if ( !pInput )
        return unexpected( "No primary input for the'" + _name + "' element" );

    const size_t inputTaskCount = pInput->GetTaskCount();
    if ( inputTaskCount == 0 )
        return unexpected( "No input frames for the'" + _name + "' element" );

    try
    {
        auto pBitmap = pInput->RunTaskAndReportProgress( 0 );
        if ( !pBitmap )
            return unexpected( pBitmap.error() );

        std::shared_ptr<BaseStacker> pStacker = MainWindow::GetInstance( FontRegistry::Instance() ).isCudaEnabled() ? std::shared_ptr<BaseStacker>(new cuda::Stacker(**pBitmap, _stackMode)) :
            std::shared_ptr<BaseStacker>( new Stacker( **pBitmap, _stackMode ) );

        pStacker->SetThreshold( _threshold );
        pStacker->AddBitmap( *pBitmap );
        _taskReadiness = 1.0f / (inputTaskCount + 1);

        for ( size_t i = 1; i < inputTaskCount; ++i )
        {
            pBitmap = pInput->RunTaskAndReportProgress( i );
            if ( !pBitmap )
                return unexpected( pBitmap.error() );

            pStacker->AddBitmap( *pBitmap );

            _taskReadiness = float( i ) / (inputTaskCount + 1);
        }

        auto res = pStacker->GetResult();
        _completedTaskCount = 1;
        _taskReadiness = 0.0f;
        return res;
    }
    catch ( std::exception& e )
    {
        return unexpected( e.what() );
    }
}

void StackerWindow::Serialize( std::ostream& out ) const
{
    PipelineElementWindow::Serialize( out );
    gui::Serialize( _stackMode, out );
    gui::Serialize( _threshold, out );
}

void StackerWindow::Deserialize( std::istream& in )
{
    PipelineElementWindow::Deserialize( in );
    _stackMode = gui::Deserialize<StackMode>( in, _remainingBytes );
    _threshold = gui::Deserialize<float>( in, _remainingBytes );
}

int StackerWindow::GetSerializedStringSize() const
{
    return PipelineElementWindow::GetSerializedStringSize()
        + gui::GetSerializedStringSize( _stackMode )
        + gui::GetSerializedStringSize( _threshold );
}

void StackerWindow::ResetTasks()
{
    _completedTaskCount = 0;
    _taskReadiness = 0;
}

REGISTER_TOOLS_ITEM( StackerWindow )

ACMB_GUI_NAMESPACE_END
