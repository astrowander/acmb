#include "StackerWindow.h"
#include "MainWindow.h"
#include "Serializer.h"
#include "./../Registrator/stacker.h"
#include "./../Cuda/CudaInfo.h"
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
    ImGui::RadioButton( "Light Frames", ( int* ) (&_stackMode), int( StackMode::Light ) );
    ImGui::RadioButton( "Dark/Flat Frames", ( int* ) (&_stackMode), int( StackMode::DarkOrFlat ) );

    ImGui::Text( "Star Detection Threshold" );
    if ( _stackMode == StackMode::Light )
        ImGui::DragFloat( "##StarDetectionThreshold", &_threshold, 0.1f, 0.0f, 100.0f );

    //ImGui::Checkbox( "Enable CUDA", &_enableCuda );
}

std::expected<IBitmapPtr, std::string> StackerWindow::RunTask( size_t i )
{
    _completedTaskCount = 0;

    auto pInput = GetPrimaryInput();
    if ( !pInput )
        return std::unexpected( "No input element" );

    const size_t inputTaskCount = pInput->GetTaskCount();
    if ( inputTaskCount == 0 )
        return std::unexpected( "No input frames" );

    try
    {
        auto pBitmap = pInput->RunTaskAndReportProgress( 0 );
        if ( !pBitmap )
            return std::unexpected( pBitmap.error() );

        std::shared_ptr<BaseStacker> pStacker = cuda::isCudaAvailable() ? std::shared_ptr<BaseStacker>( new cuda::Stacker( **pBitmap, _stackMode ) ) :
            std::shared_ptr<BaseStacker>( new Stacker( **pBitmap, _stackMode ) );

        pStacker->SetThreshold( _threshold );
        pStacker->AddBitmap( *pBitmap );
        _taskReadiness = 1.0f / (inputTaskCount + 1);

        for ( size_t i = 1; i < inputTaskCount; ++i )
        {
            pBitmap = pInput->RunTaskAndReportProgress( i );
            if ( !pBitmap )
                return std::unexpected( pBitmap.error() );

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
        return std::unexpected( e.what() );
    }
}

void StackerWindow::Serialize( std::ostream& out )
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

int StackerWindow::GetSerializedStringSize()
{
    return PipelineElementWindow::GetSerializedStringSize()
        + gui::GetSerializedStringSize( _stackMode )
        + gui::GetSerializedStringSize( _threshold );
}

REGISTER_TOOLS_ITEM( StackerWindow )

ACMB_GUI_NAMESPACE_END
