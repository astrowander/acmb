#include "client.h"
#include <iostream>
#include <chrono>
#include <thread>

int main()
{
    try
    {
        acmb::client::Client client("127.0.0.1");
        client.Connect();
        std::cout << client.portNumber() << std::endl;
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
