#include "ImageWriterWindow.h"
#include "MainWindow.h"
#include "FileDialog.h"
#include "Serializer.h"
#include "ImGuiHelpers.h"

#include "./../Core/bitmap.h"
#include "./../Codecs/imageencoder.h"
#include "./../Codecs/H265/H265Encoder.h"
#include "./../Codecs/Y4M/Y4MEncoder.h"
#include "./../Codecs/Ser/SerEncoder.h"
#include "./../Transforms/converter.h"
#include "./../Transforms/ResizeTransform.h"

#include <sstream>

ACMB_GUI_NAMESPACE_BEGIN

static std::string GetFilters()
{
    const auto extensions = ImageEncoder::GetAllExtensions();
    std::ostringstream ss;
    ss << ".*";
    for ( const auto& extension : extensions )
    {
        ss << "," << extension;
    }
    return ss.str();
}

static std::string GetFormatList( bool excludeVideoFormats = false )
{
    const auto extensions = ImageEncoder::GetAllExtensions();
    std::ostringstream ss;
    for ( const auto& extension : extensions )
    {
        if ( excludeVideoFormats && (H265Encoder::GetExtensions().contains( extension ) || Y4MEncoder::GetExtensions().contains( extension )) )
            continue;

        ss << extension << '\0';
    }

    const auto res = ss.str();
    auto endIt = res.end();
    std::advance( endIt, -1 );
    return { res.begin(), endIt };
}

ImageWriterWindow::ImageWriterWindow( const Point& gridPos )
    : PipelineElementWindow( "Export Images", gridPos, PEFlags_StrictlyOneInput | PEFlags_NoOutput )
{
    _formatList = GetFormatList( false );
}

void ImageWriterWindow::DrawPipelineElementControls()
{
    UI::Checkbox( "Keep Original Name", &_keepOriginalFileName,
                  "If checked then the images will be saved with their original names. Otherwise the custom name will be used (with number postfixes for multiple images)" );
    ImGui::Separator();

    const float itemWidth = 100.0f * cMenuScaling;
    auto fileDialog = FileDialog::Instance();

    if ( _keepOriginalFileName )
    {
        ImGui::Text( "Directory Path" );
        ImGui::SetNextItemWidth( itemWidth );
        ImGui::InputText( "##Directory", ( char* ) _workingDirectory.c_str(), 1024, ImGuiInputTextFlags_ReadOnly );
        UI::Button( "Select Directory", { itemWidth, 0 }, [&]
        {
            fileDialog.OpenDialog( "SelectOutputFile", "Select Directory", nullptr, _workingDirectory.c_str(), 0 );
        }, "Choose a directory to export the results" );

        UI::Combo( "Format", &_formatId, _formatList, "Choose a file format to export the results" );
    }
    else
    {
        ImGui::Text( "File Name" );
        ImGui::SetNextItemWidth( itemWidth );
        const size_t lastSlash = _fileName.find_last_of( "\\/" );
        ImGui::InputText( "##File Name", ( char* ) _fileName.substr( lastSlash + 1 ).c_str(), 1024, ImGuiInputTextFlags_ReadOnly );

        UI::Button( "Select File", { itemWidth, 0 }, [&]
        {
            static auto filters = GetFilters();
            fileDialog.OpenDialog( "SelectOutputFile", "Select File", filters.c_str(), _workingDirectory.c_str(), 0 );
        }, "Choose a file format to export the results" );
    }


    if ( H265Encoder::GetExtensions().contains( _extension ) || Y4MEncoder::GetExtensions().contains( _extension ) )
    {
        UI::DragInt( "Frame Rate", &_frameRate, 0.1f, 1, 144, "Frame rate of the output video" );
        if ( H265Encoder::GetExtensions().contains( _extension ) )
            UI::DragInt( "Quality", &_quality, 0.1f, 1, 9, "1 - worst quality, fastest speed\n9 - best quality, slowest speed" );
    }

    if ( fileDialog.Display( "SelectOutputFile", {}, { 300 * cMenuScaling, 200 * cMenuScaling } ) )
    {
        _workingDirectory = fileDialog.GetCurrentPath() + "/";
        // action if OK
        if ( fileDialog.IsOk() )
        {
            _fileName = fileDialog.GetFilePathName();
            const size_t dotPos = _fileName.find_last_of( '.' );
            if ( dotPos != std::string::npos )
            {
                _extension = _fileName.substr( dotPos );
            }
        }

        // close
        fileDialog.Close();
    }

    if ( _pResultTexture )
    {
        ImGui::OpenPopup( "PreviewResult" );
    }

    if ( ImGui::BeginPopup( "PreviewResult" ) )
    {
        ImGui::Image( _pResultTexture->GetTexture(), { float( _pResultTexture->GetWidth() ), float( _pResultTexture->GetHeight() ) } );
        if ( ImGui::IsKeyPressed( ImGuiKey_Escape ) )
        {
            _pResultTexture.reset();
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
}

void ImageWriterWindow::Serialize( std::ostream& out ) const
{
    PipelineElementWindow::Serialize( out );
    gui::Serialize( _workingDirectory, out );
    gui::Serialize( _fileName, out );
    gui::Serialize( _formatId, out );
    gui::Serialize( _keepOriginalFileName, out );
    gui::Serialize( _frameRate, out );
    gui::Serialize( _quality, out );
}

void ImageWriterWindow::Deserialize( std::istream& in )
{
    PipelineElementWindow::Deserialize( in );
    _workingDirectory = gui::Deserialize<std::string>( in, _remainingBytes );
    _fileName = gui::Deserialize<std::string>( in, _remainingBytes );
    _formatId = gui::Deserialize<int>( in, _remainingBytes );
    _keepOriginalFileName = gui::Deserialize<bool>( in, _remainingBytes );
    _frameRate = gui::Deserialize<int>( in, _remainingBytes );
    _quality = gui::Deserialize<int>( in, _remainingBytes );

    const size_t dotPos = _fileName.find_last_of( '.' );
    if ( dotPos != std::string::npos )
    {
        _extension = _fileName.substr( dotPos );
    }
}

int ImageWriterWindow::GetSerializedStringSize() const
{
    return PipelineElementWindow::GetSerializedStringSize()
        + gui::GetSerializedStringSize( _workingDirectory )
        + gui::GetSerializedStringSize( _fileName )
        + gui::GetSerializedStringSize( _formatId )
        + gui::GetSerializedStringSize( _keepOriginalFileName )
        + gui::GetSerializedStringSize( _frameRate )
        + gui::GetSerializedStringSize( _quality );
}

IBitmapPtr ImageWriterWindow::ProcessBitmapFromPrimaryInput( IBitmapPtr pSource, size_t taskNumber )
{
    if ( _pEncoder )
    {
        _pEncoder->WriteBitmap( pSource );
        return nullptr;
    }

    std::string finalName;
    if ( _keepOriginalFileName )
    {
        auto taskName = GetTaskName( taskNumber );
        const size_t lastSlashPos = taskName.find_last_of( "\\/" );
        const size_t lastDotPos = taskName.find_last_of( '.' );

        static const auto extensions = ImageEncoder::GetAllExtensions();
        auto extensionIt = extensions.begin();
        std::advance( extensionIt, _formatId );

        finalName = _workingDirectory + "/" + taskName.substr( lastSlashPos + 1, lastDotPos - lastSlashPos - 1 ) + *extensionIt;
    }
    else
    {
        const size_t dotPos = _fileName.find_last_of( '.' );
        std::size_t numberSize = 0;
        const auto taskCount = GetTaskCount();
        if ( taskCount >= 1000 )
            numberSize = 4;
        else if ( taskCount >= 100 )
            numberSize = 3;
        else if ( taskCount >= 10 )
            numberSize = 2;
        else
            numberSize = 1;

        std::string numberWithLeadingZeros( numberSize, '0' );
        std::string number = std::to_string( taskNumber );
        numberWithLeadingZeros.replace( numberWithLeadingZeros.size() - number.size(), number.size(), number );

        finalName = (taskNumber == 0) ? _fileName : (_fileName.substr( 0, dotPos ) + "_" + numberWithLeadingZeros + _fileName.substr( dotPos ));
    }

    if ( taskNumber == 0 && GetTaskCount() == 1 )
    {
        auto pResized = ResizeTransform::Resize( pSource, ResizeTransform::GetSizeWithPreservedRatio( { int( pSource->GetWidth() ), int( pSource->GetHeight() ) }, { 1280, 720 } ) );
        _pResultTexture = std::make_unique<Texture>( std::static_pointer_cast< Bitmap<PixelFormat::RGBA32> >(Converter::Convert( pResized, PixelFormat::RGBA32 )) );
    }
    IBitmap::Save( pSource, finalName );
    return nullptr;
}

std::vector<std::string> ImageWriterWindow::ExportAllImages()
{
    auto pPrimaryInput = GetPrimaryInput();
    if ( !pPrimaryInput )
        return { "No primary input for the'" + _name + "' element" };

    ResetProgress( PropagationDir::Backward );

    _taskCount = pPrimaryInput->GetTaskCount();
    if ( _taskCount == 0 )
        return { "No input frames for the'" + _name + "' element" };

    if ( _keepOriginalFileName && _workingDirectory.empty() )
        return { "Working directory for the'" + _name + "' element is not specified" };

    if ( !_keepOriginalFileName && _fileName.empty() )
        return { "File name for the'" + _name + "' element is not specified" };

    const bool isH265 = H265Encoder::GetExtensions().contains( _extension );
    const bool isY4M = Y4MEncoder::GetExtensions().contains( _extension );
    const bool isSER = SerEncoder::GetExtensions().contains( _extension );
    const bool isVideo = isH265 || isY4M || isSER;
    if ( isVideo )
    {
        if ( isH265 )
            _pEncoder = std::make_shared<H265Encoder>( H265Encoder::Preset( _quality - 1 ), H265Encoder::Tune::Animation );
        else if ( isY4M )
            _pEncoder = std::make_shared<Y4MEncoder>();
        else
            _pEncoder = std::make_shared<SerEncoder>();
        
        _pEncoder->SetFrameRate( _frameRate );
        _pEncoder->Attach( _fileName );
    }

    std::vector<std::string> res;
    for ( size_t i = 0; i < _taskCount; ++i )
    {
        const auto taskRes = RunTaskAndReportProgress( i );
        if ( !taskRes.has_value() )
        {
            res.push_back( taskRes.error() );
            ResetProgress( PropagationDir::Backward );
            break;
        }
    }

    if ( _pEncoder )
    {
        if ( isVideo )
            _pEncoder->SetTotalFrames( uint32_t( _taskCount ) );

        _pEncoder->Detach();
        _pEncoder.reset();
    }
    return res;
}

Expected<void, std::string> ImageWriterWindow::GeneratePreviewBitmap()
{
    _pPreviewBitmap = GetPrimaryInput()->GetPreviewBitmap();
    return {};
}

REGISTER_TOOLS_ITEM( ImageWriterWindow )

ACMB_GUI_NAMESPACE_END
