#include "MainWindow.h"
#include "ImageReaderWindow.h"
#include "ImageWriterWindow.h"

#include <thread>

ACMB_GUI_NAMESPACE_BEGIN

MainWindow::MainWindow( const ImVec2& pos, const ImVec2& size )
: Window( "acmb", pos, size )
{
    
}

void MainWindow::DrawDialog()
{
    if ( _children.empty() )
    {
        auto pReader = std::shared_ptr<ImageReaderWindow>( new ImageReaderWindow( ImVec2{ 0, 0 }, shared_from_this() ) );
        auto pWriter = std::shared_ptr<ImageWriterWindow>( new ImageWriterWindow( ImVec2{ 500, 0 }, shared_from_this() ) );
        pWriter->SetPrimaryInput( pReader );
        _writers.push_back( pWriter );

        _children.push_back( pReader );
        _children.push_back( pWriter );
    }
    
    if ( ImGui::Button( "Run" ) )
    {
        _errors.clear();
        _finished = false;
        std::thread process( [&]
        {
            for ( auto pWriter : _writers )
            {
                const auto errors = pWriter.lock()->RunAllTasks();
                _errors.insert( _errors.end(), errors.begin(), errors.end() );
            }

            _finished = true;
        } );
        process.detach();        
    }

    if ( _finished )
    {
        ImGui::OpenPopup( "ResultsPopup" );
        _finished = false;
    }

    if ( ImGui::BeginPopup( "ResultsPopup" ) )
    {
        if ( _errors.empty() )
        {
            ImGui::TextColored( { 0, 1, 0, 1 }, "Success!" );
            return ImGui::EndPopup();
        }

        for ( const auto& error : _errors )
        {
            ImGui::TextColored( { 1, 0, 0, 1 }, "%s", error.c_str() );
        }

        return ImGui::EndPopup();
    }
}

std::shared_ptr<MainWindow> MainWindow::Create()
{
    //const auto viewport = ImGui::GetMainViewport();
    return std::shared_ptr<MainWindow>( new MainWindow( { 0, 0 }, { 1280, 800 } ) );
}

ACMB_GUI_NAMESPACE_END


