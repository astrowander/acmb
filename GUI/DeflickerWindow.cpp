#include "DeflickerWindow.h"
#include "MainWindow.h"
#include "Serializer.h"
#include "ImGuiHelpers.h"

#include "./../Transforms/DeflickerTransform.h"
ACMB_GUI_NAMESPACE_BEGIN

DeflickerWindow::DeflickerWindow( const Point& gridPos )
: PipelineElementWindow( "Deflicker", gridPos, PEFlags_StrictlyOneInput | PEFlags_StrictlyOneOutput )
{
    _bitmaps.resize( _framesPerChunk );
}

void DeflickerWindow::DrawPipelineElementControls()
{
    ImGui::Text( "Frames per Chunk:" );
    UI::DragInt( "##Frames per Chunk", &_framesPerChunk, 1.0f, 1, 100, "Number of frames per chunk", this );
    if ( _framesPerChunk != _bitmaps.size() )
        _bitmaps.resize( _framesPerChunk );
}

Expected<IBitmapPtr, std::string> DeflickerWindow::RunTask( size_t i )
{
    auto pInput = GetPrimaryInput();
    if ( !pInput )
        return unexpected( "No primary input for the'" + _name + "' element" );

    if ( i % _framesPerChunk == 0 )
    {
        for ( size_t j = 0; j < std::min<size_t>( _framesPerChunk, pInput->GetTaskCount() - i ); ++j )
        {
            auto pBitmap = pInput->RunTaskAndReportProgress( i + j );
            if ( !pBitmap )
                return unexpected( pBitmap.error() );

            _bitmaps[j]= *pBitmap;
        }

        DeflickerTransform::Settings settings { .bitmaps = _bitmaps, .iterations = _iterations };

        DeflickerTransform::Deflicker( settings );
    }

    return _bitmaps[i % _framesPerChunk];
}

void DeflickerWindow::Serialize( std::ostream& out ) const
{
    PipelineElementWindow::Serialize( out );
    gui::Serialize( _framesPerChunk, out );
}

bool DeflickerWindow::Deserialize( std::istream& in )
{
    if ( !PipelineElementWindow::Deserialize( in ) ) return false;
    _framesPerChunk = gui::Deserialize<int>( in, _remainingBytes );
    return true;
}

int DeflickerWindow::GetSerializedStringSize() const
{
    return PipelineElementWindow::GetSerializedStringSize()
        + gui::GetSerializedStringSize( _framesPerChunk );
}

Expected<void, std::string> DeflickerWindow::GeneratePreviewBitmap()
{
    _pPreviewBitmap = GetPrimaryInput()->GetPreviewBitmap();
    return {};
}

REGISTER_TOOLS_ITEM( DeflickerWindow )

ACMB_GUI_NAMESPACE_END