#include "client.h"
#include <iostream>
#include <chrono>
#include <thread>

int main( int argc, const char** argv )
{
    try
    {
        if ( argc < 3 )
            return 0;

        acmb::client::Client client("127.0.0.1");
        client.Connect();
        std::cout << client.portNumber() << std::endl;\
        client.Process( argv[1], argv[2] );
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        client.Disconnect();
        std::cout << client.portNumber() << std::endl;
    }
    catch( std::exception& e )
    {
        std::cout << "Error: " << e.what() << std::endl;
    }

    return 0;
}
