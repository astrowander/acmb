#include "LevelsWindow.h"
#include "./../Transforms/HistogramBuilder.h"
#include "./../Transforms/converter.h"

ACMB_GUI_NAMESPACE_BEGIN

LevelsWindow::LevelsWindow( const Point& gridPos )
: PipelineElementWindow( "Levels", gridPos, PEFlags_StrictlyOneInput | PEFlags_StrictlyOneOutput )
{
    auto pPrimaryInput = GetPrimaryInput();
    if ( !pPrimaryInput )
        return;

    auto pHistogramBuilder = HistorgamBuilder::Create( _pCachedPreview );
    const auto pixelFormat = _pCachedPreview->GetPixelFormat();
    const auto colorSpace = GetColorSpace( pixelFormat );
    if ( colorSpace == ColorSpace::Gray )
    {
        _channelHistograms[0] = pHistogramBuilder->GetChannelHistogram( 0 );
    }
    else if ( colorSpace == ColorSpace::RGB )
    {
        auto pGrayBitmap = Converter::Convert( _pCachedPreview, BytesPerChannel( pixelFormat ) == 1 ? PixelFormat::Gray8 : PixelFormat::Gray16 );
        auto pLumaHistogramBuilder = HistorgamBuilder::Create( pGrayBitmap );
        _channelHistograms[0] = pLumaHistogramBuilder->GetChannelHistogram( 0 );

        _channelHistograms[1] = pHistogramBuilder->GetChannelHistogram( 0 );
        _channelHistograms[2] = pHistogramBuilder->GetChannelHistogram( 1 );
        _channelHistograms[3] = pHistogramBuilder->GetChannelHistogram( 2 );
    }
}


ACMB_GUI_NAMESPACE_END