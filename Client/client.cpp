#include "client.h"
#include <boost/array.hpp>

using boost::asio::ip::tcp;

ACMB_CLIENT_NAMESPACE_BEGIN

Client::Client(const std::string& serverAddress)
:serverAddress_(serverAddress)
{
}

void Client::Connect()
{
    const auto ipAddr = boost::asio::ip::address::from_string( serverAddress_ );
    boost::array< tcp::endpoint, 1> endpoints = { tcp::endpoint( ipAddr, cHelloPort ) };

    tcp::socket socket_( context_ );
    boost::asio::connect(socket_, endpoints );

    boost::array<int, 2> command = { 1, 0 };
    boost::array<int, 1> answer = {};

    boost::system::error_code error;
    for (;;)
    {
        socket_.write_some(boost::asio::buffer(command), error);
        socket_.read_some(boost::asio::buffer(answer), error);
        if (error == boost::asio::error::eof)
          break; // Connection closed cleanly by peer.
        else if (error)
          throw boost::system::system_error(error); // Some other error.

        if ( answer[0] == -1)
            throw std::runtime_error("Unable to connect");

        portNumber_ = answer[0];
    }
}

void Client::Disconnect()
{
    const auto ipAddr = boost::asio::ip::address::from_string( serverAddress_ );
    boost::array< tcp::endpoint, 1> endpoints = { tcp::endpoint( ipAddr, cHelloPort ) };

    tcp::socket socket_( context_ );
    boost::asio::connect(socket_, endpoints );

    boost::array<int, 2> command = { 2, portNumber_ };
    boost::array<int, 1> answer = {};

    boost::system::error_code error;
    for (;;)
    {
        socket_.write_some(boost::asio::buffer(command), error);
        socket_.read_some(boost::asio::buffer(answer), error);
        if (error == boost::asio::error::eof)
          break; // Connection closed cleanly by peer.
        else if (error)
          throw boost::system::system_error(error); // Some other error.

        if ( answer[0] == -1)
            throw std::runtime_error("Unable to disconnect");
    }
    portNumber_ = -1;
}

ACMB_CLIENT_NAMESPACE_END
