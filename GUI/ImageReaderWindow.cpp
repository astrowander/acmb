#include "ImageReaderWindow.h"
#include "MainWindow.h"
#include "Serializer.h"
#include "FileDialog.h"
#include "ImGuiHelpers.h"

#include "./../Codecs/imagedecoder.h"
#include "./../Transforms/converter.h"

#include <sstream>

ACMB_GUI_NAMESPACE_BEGIN

std::string GetFilters()
{
    const auto extensions = ImageDecoder::GetAllExtensions();
    std::ostringstream ss;
    ss << ".*";
    for ( const auto& extension : extensions )
    {
        ss << "," << extension;
    }
    return ss.str();
}

ImageReaderWindow::ImageReaderWindow( const Point& gridPos )
: PipelineElementWindow( "Import Images", gridPos, PEFlags_NoInput | PEFlags_StrictlyOneOutput )
, _workingDirectory( "." )
{
}

void ImageReaderWindow::DrawPipelineElementControls()
{
    const auto& style = ImGui::GetStyle();
    const float itemWidth = cElementWidth - 2.0f * style.WindowPadding.x;

    if ( ImGui::BeginListBox( "##ImageList", { itemWidth, 100 } ) )
    {
        for ( int i = 0; i < int( _fileNames.size() ); ++i )
        {
            const bool is_selected = ( _selectedItemIdx == i );
            const std::string shortName = _fileNames[i].substr( _fileNames[i].find_last_of( "\\/" ) + 1 );
            if ( ImGui::Selectable( shortName.c_str(), is_selected ) )
                _selectedItemIdx = i;

            // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
            if ( is_selected )
                ImGui::SetItemDefaultFocus();
        }
        ImGui::EndListBox();
    }

    UI::Button( "Show Preview", { itemWidth, 0 }, [&]
    {
        if ( _selectedItemIdx >= _fileNames.size() )
            return;

        auto pDecoder = ImageDecoder::Create( _fileNames[_selectedItemIdx] );
        _pPreviewTexture = std::make_unique<Texture>( std::static_pointer_cast<Bitmap<PixelFormat::RGBA32>>(Converter::Convert( pDecoder->ReadPreview(), PixelFormat::RGBA32 ) ) );
    }, "Show a preview of the selected image" );

    auto fileDialog = FileDialog::Instance();
    const auto openDialogName = "SelectImagesDialog##" + _name;

    UI::Button( "Select Images", { itemWidth, 0 }, [&]
    {
        static auto filters = GetFilters();
        fileDialog.OpenDialog( openDialogName, "Select Images", filters.c_str(), _workingDirectory.c_str(), 0 );
        ResetProgress( PropagationDir::Forward );
    }, "Add images to the importing list" );

    UI::Button( "Clear List", { itemWidth, 0 }, [&]
    {
        _fileNames.clear();
        _selectedItemIdx = 0;
        ResetProgress( PropagationDir::Forward );
    }, "Delete all images from the importing list" );

    if ( fileDialog.Display( openDialogName, {}, { 300 * cMenuScaling, 200 * cMenuScaling } ) )
    {
        // action if OK
        if ( fileDialog.IsOk() )
        {
            _workingDirectory = fileDialog.GetCurrentPath();

            const auto selection = fileDialog.GetSelection();
            for ( const auto& it : selection )
                _fileNames.push_back( _workingDirectory + "/" + it.first);
        }

        // close
        fileDialog.Close();
    }

    auto& mainWindow = MainWindow::GetInstance( FontRegistry::Instance() );

    if ( _pPreviewTexture )
    {
        ImGui::OpenPopup( "Preview" );
        mainWindow.LockInterface();
    }

    if ( ImGui::BeginPopup( "Preview" ) )
    {
        ImGui::Image( _pPreviewTexture->GetTexture(), { float( _pPreviewTexture->GetWidth() ), float( _pPreviewTexture->GetHeight() ) } );
        if ( ImGui::IsKeyPressed( ImGuiKey_Escape ) )
        {
            mainWindow.UnlockInterface();
            _pPreviewTexture.reset();
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
}

Expected<IBitmapPtr, std::string> ImageReaderWindow::RunTask( size_t i )
{
    try
    {
        return IBitmap::Create( _fileNames[i] );
    }
    catch ( std::exception& e )
    {
        return unexpected( e.what() );
    }
}

void ImageReaderWindow::Serialize(std::ostream& out) const
{
    PipelineElementWindow::Serialize(out);
    gui::Serialize( _workingDirectory, out);
    gui::Serialize( std::move( _fileNames ), out);
    gui::Serialize(_selectedItemIdx, out);    
}

void ImageReaderWindow::Deserialize(std::istream& in)
{
    PipelineElementWindow::Deserialize(in);
    _workingDirectory = gui::Deserialize<std::string>(in, _remainingBytes);
    _fileNames = gui::Deserialize<std::vector<std::string>>(in, _remainingBytes);
    _selectedItemIdx = gui::Deserialize<int>(in, _remainingBytes);
}

int ImageReaderWindow::GetSerializedStringSize() const
{
    return PipelineElementWindow::GetSerializedStringSize() 
        + gui::GetSerializedStringSize( _workingDirectory )
        + gui::GetSerializedStringSize( _fileNames )
        + gui::GetSerializedStringSize( _selectedItemIdx );
}

std::string ImageReaderWindow::GetTaskName( size_t taskNumber ) const
{
    return (taskNumber < _fileNames.size()) ? _fileNames[taskNumber] : std::string{};
}

size_t ImageReaderWindow::GetTaskCount()
{
    return _fileNames.size();
}

REGISTER_TOOLS_ITEM( ImageReaderWindow )

ACMB_GUI_NAMESPACE_END
