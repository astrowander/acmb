#include "DarkFrameWindow.h"
#include "MainWindow.h"
#include "Serializer.h"
#include "../Transforms/BitmapSubtractor.h"
#include "../Transforms/HistogramBuilder.h"

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/enumerable_thread_specific.h>

#include "ImGuiHelpers.h"

ACMB_GUI_NAMESPACE_BEGIN

DarkFrameWindow::DarkFrameWindow( const Point& gridPos )
: PipelineElementWindow( "Subtract Dark Frame", gridPos, PEFlags::PEFlags_StrictlyTwoInputs | PEFlags::PEFlags_StrictlyOneOutput )
{
}

void DarkFrameWindow::DrawPipelineElementControls()
{
    UI::DragFloat( "Multiplier", &_multiplier, 0.001f, 0.2f, 5.0f, "Each pixel of the dark frame will be multiplied by this factor before subtracting", this );
    UI::Button( "Adjust Multiplier", { -1, 0 }, [&]
    {
        auto& mainWindow = MainWindow::GetInstance( FontRegistry::Instance() );
        mainWindow.LockInterface();
        auto res = AutoAdjustMultiplier();
        mainWindow.UnlockInterface();
        if ( !res )
        {
            _showError = true;
            _error = res.error();
            return;
        }
        _multiplier = res.value();
    }, "Calculate appropriate multiplier automatically", this );
    
    ImGui::Separator();
    if ( ImGui::CollapsingHeader( "Advanced" ) )
    {
        UI::Checkbox( "Dark Frame is on Left", &_primaryInputIsOnTop, "By default the top image is subtracted from the left one. If checked, the left image is subtracted from the top one", this );
    }
}

IBitmapPtr DarkFrameWindow::ProcessBitmapFromPrimaryInput( IBitmapPtr pSource, size_t )
{
    return BitmapSubtractor::Subtract( pSource, { .pBitmapToSubtract = _pSecondaryInputResult, .multiplier = _multiplier } );
}

void DarkFrameWindow::Serialize( std::ostream& out ) const
{
    PipelineElementWindow::Serialize( out );
    gui::Serialize( _multiplier, out );
}

int DarkFrameWindow::GetSerializedStringSize() const
{
    return PipelineElementWindow::GetSerializedStringSize() + gui::GetSerializedStringSize( _multiplier );
}

bool DarkFrameWindow::Deserialize( std::istream& in )
{
    if ( !PipelineElementWindow::Deserialize( in ) ) return false;
    _multiplier = gui::Deserialize<float>( in, _remainingBytes );
    return true;
}

template<PixelFormat pixelFormat>
float CalculateIntensity( std::shared_ptr<Bitmap<pixelFormat>> pSource, std::shared_ptr<Bitmap<pixelFormat>> pDarkFrame )
{    
    using ChannelType = typename PixelFormatTraits<pixelFormat>::ChannelType;
    constexpr size_t channelCount = PixelFormatTraits<pixelFormat>::channelCount;

    auto pHistogramBuilder = HistogramBuilder::Create( pDarkFrame );
    pHistogramBuilder->BuildHistogram();
    std::vector<uint32_t> thresholds( channelCount );    

    //// Calculate thresholds finding 100 most bright pixels in each channel of dark frame
    for ( uint32_t ch = 0; ch < channelCount; ++ch )
    {
        const auto& histogram = pHistogramBuilder->GetChannelHistogram( ch );
        const auto& statistics = pHistogramBuilder->GetChannelStatistics( ch );

        uint32_t count = 0;

        for ( int32_t val = statistics.max; val >= 0; --val )
        {
            count += histogram[val];
            if ( count >= 100 )
            {
                thresholds[ch] = val;
                break;
            }
        }
    }

    tbb::enumerable_thread_specific<std::vector<float>> multipliers;

    auto sourceBlackLevel = pSource->GetCameraSettings()->blackLevel;
    auto darkBlackLevel = pDarkFrame->GetCameraSettings()->blackLevel;

    tbb::parallel_for( tbb::blocked_range<uint32_t>( 0, pSource->GetHeight() ), [&] ( const tbb::blocked_range<uint32_t>& range )
    {
        auto& local = multipliers.local();

        for ( uint32_t i = range.begin(); i < range.end(); ++i )
        {
            auto scanline = pSource->GetScanline( i );
            auto darkScanline = pDarkFrame->GetScanline( i );

            for ( size_t j = 0; j < pSource->GetWidth(); ++j )
            {
                for ( size_t ch = 0; ch < channelCount; ++ch )
                {
                    const auto darkValue = *darkScanline;
                    if ( darkValue >= thresholds[ch] )
                    {
                        const auto value = *scanline;
                        const auto multiplier = float( value - sourceBlackLevel ) / float( darkValue - darkBlackLevel );
                        local.push_back( multiplier );
                    }

                    ++scanline;
                    ++darkScanline;
                }
            }
        }
    } );

    float result = 0.0f;
    size_t count = 0;
    for ( const auto& multipliersPerThread : multipliers )
    {
        for ( const float multiplier : multipliersPerThread )
        {
            result += multiplier;
            ++count;
        }
    }

    return result / count;
}

Expected<float, std::string> DarkFrameWindow::AutoAdjustMultiplier()
{
    auto pPrimaryInput = GetPrimaryInput();
    auto pSecondaryInput = GetSecondaryInput();
    if ( !pPrimaryInput || pPrimaryInput->GetTaskCount() == 0 || !pSecondaryInput  || pSecondaryInput->GetTaskCount() == 0 )
        return unexpected( "no input element" );

    auto pDarkFrameExp = pSecondaryInput->RunTaskAndReportProgress( 0 );
    if ( !pDarkFrameExp )
        return unexpected( "Dark frame processing error" );

    auto pSourceExp = pPrimaryInput->RunTaskAndReportProgress( 0 );
    if ( !pSourceExp )
        return unexpected( "Light frame processing error" );
    
    pPrimaryInput->ResetProgress( PipelineElementWindow::PropagationDir::Backward );
    pSecondaryInput->ResetProgress( PipelineElementWindow::PropagationDir::Backward );

    auto pSource = pSourceExp.value();
    auto pDarkFrame = pDarkFrameExp.value();
    if ( pSource->GetPixelFormat() != pDarkFrame->GetPixelFormat() )
        return unexpected( "pixel format mismatch" );

    if ( pSource->GetWidth() != pDarkFrame->GetWidth() || pSource->GetHeight() != pDarkFrame->GetHeight() )
        return unexpected( "image size mismatch" );

    switch ( pDarkFrame->GetPixelFormat() )
    {
        case PixelFormat::Gray8:
            return CalculateIntensity( std::static_pointer_cast<Bitmap<PixelFormat::Gray8>>( pSource ), std::static_pointer_cast<Bitmap<PixelFormat::Gray8>>( pDarkFrame ) );            
        case PixelFormat::Gray16:
            return CalculateIntensity( std::static_pointer_cast< Bitmap<PixelFormat::Gray16> >(pSource), std::static_pointer_cast< Bitmap<PixelFormat::Gray16> >(pDarkFrame) );
        case PixelFormat::RGB24:
            return CalculateIntensity( std::static_pointer_cast< Bitmap<PixelFormat::RGB24> >(pSource), std::static_pointer_cast< Bitmap<PixelFormat::RGB24> >(pDarkFrame) );
        case PixelFormat::RGB48:
            return CalculateIntensity( std::static_pointer_cast< Bitmap<PixelFormat::RGB48> >(pSource), std::static_pointer_cast< Bitmap<PixelFormat::RGB48> >(pDarkFrame) );
        case PixelFormat::Bayer16:
            return CalculateIntensity( std::static_pointer_cast< Bitmap<PixelFormat::Bayer16> >(pSource), std::static_pointer_cast< Bitmap<PixelFormat::Bayer16> >(pDarkFrame) );
        default:
            return unexpected( "unsupported pixel format" );
    }
}

Expected<void, std::string> DarkFrameWindow::GeneratePreviewBitmap()
{    
    _pPreviewBitmap = BitmapSubtractor::Subtract( GetPrimaryInput()->GetPreviewBitmap()->Clone(), {.pBitmapToSubtract = GetSecondaryInput()->GetPreviewBitmap(), .multiplier = _multiplier});
    _pPreviewTexture = std::make_unique<Texture>( _pPreviewBitmap );
    return {};
}

REGISTER_TOOLS_ITEM( DarkFrameWindow )

ACMB_GUI_NAMESPACE_END