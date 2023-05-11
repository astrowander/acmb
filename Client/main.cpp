#include "client.h"
#include <iostream>
#include <sstream>

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

        acmb::client::Client client( argv[2] );
        client.Connect();
        std::cout << client.portNumber() << std::endl;

        std::vector<std::string> args;
        for ( int i = 3; i < argc; ++i )
            args.emplace_back( argv[i] );

        client.Process( args );

        client.Disconnect();
        std::cout << client.portNumber() << std::endl;
    }
    catch( std::exception& e )
    {
        std::cout << "Error: " << e.what() << std::endl;
    }

    return 0;
}
