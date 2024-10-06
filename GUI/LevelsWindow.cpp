#include "LevelsWindow.h"
#include "Serializer.h"
#include "MainWindow.h"
#include "ImGuiHelpers.h"

#include "./../Transforms/HistogramBuilder.h"
#include "./../Transforms/converter.h"

ACMB_GUI_NAMESPACE_BEGIN

LevelsWindow::LevelsWindow( const Point& gridPos )
: PipelineElementWindow( "Levels", gridPos, PEFlags_StrictlyOneInput | PEFlags_StrictlyOneOutput )
{   
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
    ImGui::SetNextItemWidth( colWidth );
    UI::DragFloat( "##LMin", &_levelsSettings.levels[0].min, 0.001f, 0.0f, 1.0f, "Minimum value of the input image", this);
    ImGui::TableNextColumn();
    ImGui::SetCursorPosY( ImGui::GetCursorPosY() - ImGui::GetStyle().FramePadding.y );
    ImGui::SetNextItemWidth( colWidth );
    UI::DragFloat( "##LGamma", &_levelsSettings.levels[0].gamma, 0.001f, 0.1f, 10.0f, "Gamma", this );
    ImGui::TableNextColumn();
    ImGui::SetCursorPosY( ImGui::GetCursorPosY() - ImGui::GetStyle().FramePadding.y );
    ImGui::SetNextItemWidth( colWidth );
    UI::DragFloat( "##LMax", &_levelsSettings.levels[0].max, 0.001f, 0.0f, 1.0f, "Maximum value of the input image", this );


    if ( _levelsSettings.adjustChannels )
    {
        ImGui::TableNextColumn();
        ImGui::Text( "R" );
        ImGui::TableNextColumn();

        ImGui::SetCursorPosY( ImGui::GetCursorPosY() - ImGui::GetStyle().FramePadding.y );
        ImGui::SetNextItemWidth( colWidth );
        UI::DragFloat( "##RMin", &_levelsSettings.levels[1].min, 0.001f, 0.0f, 1.0f, "Minimum value of the input image", this );
        ImGui::TableNextColumn();

        ImGui::SetCursorPosY( ImGui::GetCursorPosY() - ImGui::GetStyle().FramePadding.y );
        ImGui::SetNextItemWidth( colWidth );
        UI::DragFloat( "##RGamma", &_levelsSettings.levels[1].gamma, 0.001f, 0.1f, 10.0f, "Gamma", this );
        ImGui::TableNextColumn();

        ImGui::SetCursorPosY( ImGui::GetCursorPosY() - ImGui::GetStyle().FramePadding.y );
        ImGui::SetNextItemWidth( colWidth );
        UI::DragFloat( "##RMax", &_levelsSettings.levels[1].max, 0.001f, 0.0f, 1.0f, "Maximum value of the input image", this );
       

        ImGui::TableNextColumn();
        ImGui::Text( "G" );
        ImGui::TableNextColumn();

        ImGui::SetCursorPosY( ImGui::GetCursorPosY() - ImGui::GetStyle().FramePadding.y );
        ImGui::SetNextItemWidth( colWidth );
        UI::DragFloat( "##GMin", &_levelsSettings.levels[2].min, 0.001f, 0.0f, 1.0f, "Minimum value of the input image", this );
        ImGui::TableNextColumn();

        ImGui::SetCursorPosY( ImGui::GetCursorPosY() - ImGui::GetStyle().FramePadding.y );
        ImGui::SetNextItemWidth( colWidth );
        UI::DragFloat( "##GGamma", &_levelsSettings.levels[2].gamma, 0.001f, 0.1f, 10.0f, "Gamma", this );
        ImGui::TableNextColumn();

        ImGui::SetCursorPosY( ImGui::GetCursorPosY() - ImGui::GetStyle().FramePadding.y );
        ImGui::SetNextItemWidth( colWidth );
        UI::DragFloat( "##GMax", &_levelsSettings.levels[2].max, 0.001f, 0.0f, 1.0f, "Maximum value of the input image", this );


        ImGui::TableNextColumn();
        ImGui::Text( "B" );
        ImGui::TableNextColumn();

        ImGui::SetCursorPosY( ImGui::GetCursorPosY() - ImGui::GetStyle().FramePadding.y );
        ImGui::SetNextItemWidth( colWidth );
        UI::DragFloat( "##BMin", &_levelsSettings.levels[3].min, 0.001f, 0.0f, 1.0f, "Minimum value of the input image", this );
        ImGui::TableNextColumn();

        ImGui::SetCursorPosY( ImGui::GetCursorPosY() - ImGui::GetStyle().FramePadding.y );
        ImGui::SetNextItemWidth( colWidth );
        UI::DragFloat( "##BGamma", &_levelsSettings.levels[3].gamma, 0.001f, 0.1f, 10.0f, "Gamma", this );
        ImGui::TableNextColumn();

        ImGui::SetCursorPosY( ImGui::GetCursorPosY() - ImGui::GetStyle().FramePadding.y );
        ImGui::SetNextItemWidth( colWidth );
        UI::DragFloat( "##BMax", &_levelsSettings.levels[3].max, 0.001f, 0.0f, 1.0f, "Maximum value of the input image", this );
    }
    ImGui::EndTable();

    UI::Checkbox( "Adjust Channels", &_levelsSettings.adjustChannels, "Adjust the channels of the input image", this );
    UI::Button( "Auto Levels", { -1, 0 }, [&]
    {
        auto& mainWindow = MainWindow::GetInstance( FontRegistry::Instance() );
        mainWindow.LockInterface();
        auto res = AutoAdjustLevels();
        mainWindow.UnlockInterface();
        if ( !res )
        {
            _showError = true;
            _error = res.error();
            return;
        }
    }, "Automatically adjust levels", this );

    ImGui::PopStyleVar();
}

Expected<void, std::string> LevelsWindow::AutoAdjustLevels()
{
    auto pInputBitmap = GetPrimaryInput()->GetPreviewBitmap();
    auto pHistogramBuilder = HistogramBuilder::Create( pInputBitmap );
    pHistogramBuilder->BuildHistogram();

    const auto colorSpace = GetColorSpace( pInputBitmap->GetPixelFormat() );
    const auto pixelFormat = pInputBitmap->GetPixelFormat();
    const auto bytesPerChannel = BytesPerChannel( pixelFormat );
    const float absoluteMax = (bytesPerChannel == 1) ? 255.0f : 65535.0f;
    
    constexpr float logTargetMedian = -2.14f;

    switch ( colorSpace )
    {
        case ColorSpace::Gray:
        {
            _levelsSettings.levels[0].min = pHistogramBuilder->GetChannelStatistics( 0 ).min / absoluteMax;
            _levelsSettings.levels[0].max = pHistogramBuilder->GetChannelStatistics( 0 ).max / absoluteMax;
            const float denom = log( (pHistogramBuilder->GetChannelStatistics( 0 ).median / absoluteMax - _levelsSettings.levels[0].min) / (_levelsSettings.levels[0].max - _levelsSettings.levels[0].min) );
            _levelsSettings.levels[0].gamma = denom / logTargetMedian;
            break;
        }
        case ColorSpace::RGB:
        {
            std::array<float, 3> channelMins = { pHistogramBuilder->GetChannelStatistics( 0 ).min / absoluteMax, pHistogramBuilder->GetChannelStatistics( 1 ).min / absoluteMax, pHistogramBuilder->GetChannelStatistics( 2 ).min / absoluteMax };
            std::array<float, 3> channelMaxs = { pHistogramBuilder->GetChannelStatistics( 0 ).max / absoluteMax, pHistogramBuilder->GetChannelStatistics( 1 ).max / absoluteMax, pHistogramBuilder->GetChannelStatistics( 2 ).max / absoluteMax };
            std::array<float, 3> channelMedians = { pHistogramBuilder->GetChannelStatistics( 0 ).median / absoluteMax, pHistogramBuilder->GetChannelStatistics( 1 ).median / absoluteMax, pHistogramBuilder->GetChannelStatistics( 2 ).median / absoluteMax };
            _levelsSettings.levels[0].min = std::min( { channelMins[0], channelMins[1], channelMins[2] } );
            _levelsSettings.levels[0].max = std::max( { channelMaxs[0], channelMaxs[1], channelMaxs[2] } );
            const float denom = log ( (channelMedians[1] - channelMins[1]) / (channelMaxs[1] - channelMins[1]) );            
            _levelsSettings.levels[0].gamma = denom / logTargetMedian;

            if ( _levelsSettings.adjustChannels )
            {
                const float range = _levelsSettings.levels[0].max - _levelsSettings.levels[0].min;
                _levelsSettings.levels[1].min = (channelMins[0] - _levelsSettings.levels[0].min ) / range;
                _levelsSettings.levels[1].max = (channelMaxs[0] - _levelsSettings.levels[0].min ) / range;
                _levelsSettings.levels[2].min = (channelMins[1] - _levelsSettings.levels[0].min ) / range;
                _levelsSettings.levels[2].max = (channelMaxs[1] - _levelsSettings.levels[0].min ) / range;
                _levelsSettings.levels[3].min = (channelMins[2] - _levelsSettings.levels[0].min ) / range;
                _levelsSettings.levels[3].max = (channelMaxs[2] - _levelsSettings.levels[0].min ) / range;


                _levelsSettings.levels[1].gamma = log( (channelMedians[0] - channelMins[0]) / (channelMaxs[0] - channelMins[0]) ) / denom;
                _levelsSettings.levels[3].gamma = log( (channelMedians[2] - channelMins[2]) / (channelMaxs[2] - channelMins[2]) ) / denom;
            }
            break;
        }
        default:
            return unexpected( "unsupported pixel format" );
    }
    return {};
}

void LevelsWindow::Serialize( std::ostream& out ) const
{
    PipelineElementWindow::Serialize( out );
    gui::Serialize( _levelsSettings, out );
}

bool LevelsWindow::Deserialize( std::istream& in )
{
    if ( !PipelineElementWindow::Deserialize( in ) ) return false;
    _levelsSettings = gui::Deserialize<decltype(_levelsSettings)>( in, _remainingBytes );
    return true;
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
    _pPreviewBitmap = LevelsTransform::ApplyLevels( pInputBitmap, _levelsSettings );
    return {};
}

IBitmapPtr LevelsWindow::ProcessBitmapFromPrimaryInput( IBitmapPtr pSource, size_t )
{
    const auto colorSpace = GetColorSpace( pSource->GetPixelFormat() );
    return LevelsTransform::ApplyLevels( pSource, _levelsSettings );
}

REGISTER_TOOLS_ITEM( LevelsWindow );

ACMB_GUI_NAMESPACE_END
