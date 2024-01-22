#include "LevelsWindow.h"
#include "Serializer.h"
#include "MainWindow.h"
#include "ImGuiHelpers.h"

#include "./../Transforms/HistogramBuilder.h"
#include "./../Transforms/converter.h"
#include "./../Transforms/ChannelEqualizer.h"

ACMB_GUI_NAMESPACE_BEGIN

float ComputePixelValue( float srcVal, const LevelsWindow::LevelsSettings& settings )
{
    if ( srcVal < settings.min )
        return settings.min;
    if ( srcVal > settings.max )
        return settings.max;

    float res = (srcVal - settings.min) / (settings.max - settings.min);    
    return std::pow( res, settings.gamma );
}

LevelsWindow::LevelsWindow( const Point& gridPos )
: PipelineElementWindow( "Levels", gridPos, PEFlags_StrictlyOneInput | PEFlags_StrictlyOneOutput )
{
    /*auto pPrimaryInput = GetPrimaryInput();
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
    }*/
}

void LevelsWindow::DrawPipelineElementControls()
{
    ImGui::PushStyleVar( ImGuiStyleVar_CellPadding, ImVec2( 3.0f, 3.0f ) );

    ImGui::BeginTable( "##LevelsSettings", 4, ImGuiTableFlags_SizingFixedFit );
    
    constexpr float colWidth = 35.0f;
    ImGui::TableSetupColumn( "##Channel", ImGuiTableColumnFlags_WidthFixed, 10 );
    ImGui::TableSetupColumn( "Min", ImGuiTableColumnFlags_WidthFixed, colWidth );
    ImGui::TableSetupColumn( "Gamma", ImGuiTableColumnFlags_WidthFixed, colWidth );
    ImGui::TableSetupColumn( "Max", ImGuiTableColumnFlags_WidthFixed, colWidth );

    ImGui::TableNextColumn();
    ImGui::TableNextColumn();
    ImGui::SetCursorPosX( ImGui::GetCursorPosX() + ( colWidth - ImGui::CalcTextSize("Min").x ) * 0.5f );
    ImGui::Text( "Min" );
    ImGui::TableNextColumn();
    ImGui::SetCursorPosX( ImGui::GetCursorPosX() + (colWidth - ImGui::CalcTextSize( "Gam" ).x) * 0.5f );
    ImGui::Text( "Gam" );
    ImGui::TableNextColumn();
    ImGui::SetCursorPosX( ImGui::GetCursorPosX() + (colWidth - ImGui::CalcTextSize( "Max" ).x) * 0.5f );
    ImGui::Text( "Max" );

    ImGui::TableNextColumn();    
    ImGui::Text( "L" );
    ImGui::TableNextColumn();
    ImGui::SetCursorPosY( ImGui::GetCursorPosY() - ImGui::GetStyle().FramePadding.y );
    ImGui::SetNextItemWidth( 35.0f );
    UI::DragFloat( "##LMin", &_levelsSettings[0].min, 0.001f, 0.0f, 1.0f, "Minimum value of the input image", this);
    ImGui::TableNextColumn();
    ImGui::SetCursorPosY( ImGui::GetCursorPosY() - ImGui::GetStyle().FramePadding.y );
    ImGui::SetNextItemWidth( 35.0f );
    UI::DragFloat( "##LGamma", &_levelsSettings[0].gamma, 0.001f, 0.0f, 1.0f, "Gamma", this );
    ImGui::TableNextColumn();
    ImGui::SetCursorPosY( ImGui::GetCursorPosY() - ImGui::GetStyle().FramePadding.y );
    ImGui::SetNextItemWidth( 35.0f );
    UI::DragFloat( "##LMax", &_levelsSettings[0].max, 0.001f, 0.0f, 1.0f, "Maximum value of the input image", this );


    if ( _adjustChannels )
    {
        ImGui::TableNextColumn();
        ImGui::Text( "R" );
        ImGui::TableNextColumn();

        ImGui::SetCursorPosY( ImGui::GetCursorPosY() - ImGui::GetStyle().FramePadding.y );
        ImGui::SetNextItemWidth( 35.0f );
        UI::DragFloat( "##RMin", &_levelsSettings[1].min, 0.001f, 0.0f, 1.0f, "Minimum value of the input image", this );
        ImGui::TableNextColumn();

        ImGui::SetCursorPosY( ImGui::GetCursorPosY() - ImGui::GetStyle().FramePadding.y );
        ImGui::SetNextItemWidth( 35.0f );
        UI::DragFloat( "##LGamma", &_levelsSettings[0].gamma, 0.001f, 0.0f, 1.0f, "Gamma", this );
        ImGui::TableNextColumn();

        ImGui::SetCursorPosY( ImGui::GetCursorPosY() - ImGui::GetStyle().FramePadding.y );
        ImGui::SetNextItemWidth( 35.0f );
        UI::DragFloat( "##RMax", &_levelsSettings[1].max, 0.001f, 0.0f, 1.0f, "Maximum value of the input image", this );
       

        ImGui::TableNextColumn();
        ImGui::Text( "G" );
        ImGui::TableNextColumn();

        ImGui::SetCursorPosY( ImGui::GetCursorPosY() - ImGui::GetStyle().FramePadding.y );
        ImGui::SetNextItemWidth( 35.0f );
        UI::DragFloat( "##GMin", &_levelsSettings[2].min, 0.001f, 0.0f, 1.0f, "Minimum value of the input image", this );
        ImGui::TableNextColumn();

        ImGui::SetCursorPosY( ImGui::GetCursorPosY() - ImGui::GetStyle().FramePadding.y );
        ImGui::SetNextItemWidth( 35.0f );
        UI::DragFloat( "##GGamma", &_levelsSettings[2].gamma, 0.001f, 0.0f, 1.0f, "Gamma", this );
        ImGui::TableNextColumn();

        ImGui::SetCursorPosY( ImGui::GetCursorPosY() - ImGui::GetStyle().FramePadding.y );
        ImGui::SetNextItemWidth( 35.0f );
        UI::DragFloat( "##GMax", &_levelsSettings[2].max, 0.001f, 0.0f, 1.0f, "Maximum value of the input image", this );


        ImGui::TableNextColumn();
        ImGui::Text( "B" );
        ImGui::TableNextColumn();

        ImGui::SetCursorPosY( ImGui::GetCursorPosY() - ImGui::GetStyle().FramePadding.y );
        ImGui::SetNextItemWidth( 35.0f );
        UI::DragFloat( "##BMin", &_levelsSettings[3].min, 0.001f, 0.0f, 1.0f, "Minimum value of the input image", this );
        ImGui::TableNextColumn();

        ImGui::SetCursorPosY( ImGui::GetCursorPosY() - ImGui::GetStyle().FramePadding.y );
        ImGui::SetNextItemWidth( 35.0f );
        UI::DragFloat( "##BGamma", &_levelsSettings[3].gamma, 0.001f, 0.0f, 1.0f, "Gamma", this );
        ImGui::TableNextColumn();

        ImGui::SetCursorPosY( ImGui::GetCursorPosY() - ImGui::GetStyle().FramePadding.y );
        ImGui::SetNextItemWidth( 35.0f );
        UI::DragFloat( "##BMax", &_levelsSettings[3].max, 0.001f, 0.0f, 1.0f, "Maximum value of the input image", this );
    }
    ImGui::EndTable();

    UI::Checkbox( "Adjust Channels", &_adjustChannels, "Adjust the channels of the input image" );   
    

    ImGui::PopStyleVar();
}

void LevelsWindow::Serialize( std::ostream& out ) const
{
    PipelineElementWindow::Serialize( out );
    gui::Serialize( _levelsSettings, out );
}

void LevelsWindow::Deserialize( std::istream& in )
{
    PipelineElementWindow::Deserialize( in );
    _levelsSettings = gui::Deserialize<decltype(_levelsSettings)>( in, _remainingBytes );
}

int LevelsWindow::GetSerializedStringSize() const
{
    return PipelineElementWindow::GetSerializedStringSize() 
    + gui::GetSerializedStringSize( _levelsSettings );
}

Expected<void, std::string> LevelsWindow::GeneratePreviewBitmap()
{
    auto pInputBitmap = GetPrimaryInput()->GetPreviewBitmap()->Clone();
    const auto colorSpace = GetColorSpace( pInputBitmap->GetPixelFormat() );
    switch ( colorSpace )
    {
    case ColorSpace::Gray:
        _pPreviewBitmap = ChannelEqualizer::Equalize( pInputBitmap, 
        { 
            [&]( float val ) { return ComputePixelValue( val, _levelsSettings[0] ); } 
        } );
        break;
    case ColorSpace::RGB:
        _pPreviewBitmap = ChannelEqualizer::Equalize( pInputBitmap,
        {
            [&] ( float val ) { return ComputePixelValue( val, _levelsSettings[0] ); },
            [&] ( float val ) { return ComputePixelValue( val, _levelsSettings[0] ); },
            [&] ( float val ) { return ComputePixelValue( val, _levelsSettings[0] ); }
        } );

        if ( _adjustChannels )
        {
            _pPreviewBitmap = ChannelEqualizer::Equalize( _pPreviewBitmap,
            {
                [&] ( float val ) { return ComputePixelValue( val, _levelsSettings[1] ); },
                [&] ( float val ) { return ComputePixelValue( val, _levelsSettings[2] ); },
                [&] ( float val ) { return ComputePixelValue( val, _levelsSettings[3] ); }
            } );
        }
        break;
    default:
        return unexpected( "Unsupported color space" );
    }

    return {};
}

IBitmapPtr LevelsWindow::ProcessBitmapFromPrimaryInput( IBitmapPtr pSource, size_t taskNumber )
{
    const auto colorSpace = GetColorSpace( pSource->GetPixelFormat() );
    switch ( colorSpace )
    {
        case ColorSpace::Gray:
            return ChannelEqualizer::Equalize( pSource,
            {
                [&]( float val ) { return ComputePixelValue( val, _levelsSettings[0] ); }
            } );
        case ColorSpace::RGB:
        {
            auto res = ChannelEqualizer::Equalize( pSource, {
            {
                [&] ( float val ) { return ComputePixelValue( val, _levelsSettings[0] ); },
                [&] ( float val ) { return ComputePixelValue( val, _levelsSettings[0] ); },
                [&] ( float val ) { return ComputePixelValue( val, _levelsSettings[0] ); }
            } } );

            if ( _adjustChannels )
            {
                res = ChannelEqualizer::Equalize( res,
                {
                    [&] ( float val ) { return ComputePixelValue( val, _levelsSettings[1] ); },
                    [&] ( float val ) { return ComputePixelValue( val, _levelsSettings[2] ); },
                    [&] ( float val ) { return ComputePixelValue( val, _levelsSettings[3] ); }
                } );
            }
            return res;
            }
        default:
            return nullptr;
    }
}

REGISTER_TOOLS_ITEM( LevelsWindow );

ACMB_GUI_NAMESPACE_END