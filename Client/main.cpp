#include "client.h"
#include <iostream>
#include <chrono>

int main( int argc, const char** argv )
{
    try
    {
        if ( argc < 3 ||
             std::string(argv[1]) != "--host" ||
             std::string(argv[2]).substr(0, 2) == "--" )
        {
            std::cout << "Invalid command line" << std::endl;
            return 0;
        }

        const auto start = std::chrono::steady_clock::now();

        acmb::client::Client client( argv[2] );
        client.Connect();
        std::cout << "Connection established on port " << client.portNumber() << std::endl;

        std::vector<std::string> args;
        for ( int i = 3; i < argc; ++i )
            args.emplace_back( argv[i] );
        
        try
        {
            client.Process( args );
        }
        catch ( std::exception& e )
        {
            std::cout << "Server error: " << e.what() << std::endl;
        }

        client.Disconnect();
        if ( client.portNumber() == -1 )
            std::cout << "Disconnected successfully" << std::endl;

        const auto end = std::chrono::steady_clock::now();
        const auto ms = std::chrono::duration_cast< std::chrono::milliseconds >( end - start ).count();
        std::cout <<  ms / 1000 << "." << ms % 1000 << "s elapsed" << std::endl;
    }
    catch( std::exception& e )
    {
        std::cout << "Error: " << e.what() << std::endl;
    }

    return 0;
}
