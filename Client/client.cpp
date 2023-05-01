#include "client.h"
#include <boost/array.hpp>
#include <fstream>

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

void Client::Process(const std::string& inputFileName, const std::string& outputFileName)
{
    std::ifstream is( inputFileName, std::ios_base::in | std::ios_base::binary );
    if ( !is.is_open() )
        throw std::runtime_error( "unable to open input file" );

    std::vector<char> buf;
    is.seekg( 0, is.end );
    const size_t length = is.tellg();
    is.seekg( 0, is.beg );

    buf.resize( length );
    is.read( buf.data(), length );

    if ( !is )
        throw std::runtime_error( "unable to read the file" );

    is.close();

    const auto ipAddr = boost::asio::ip::address::from_string( serverAddress_ );
    boost::array< tcp::endpoint, 1> endpoints = { tcp::endpoint( ipAddr, portNumber_ ) };

    tcp::socket socket( context_ );
    boost::asio::connect(socket, endpoints );

    boost::array<size_t, 1> size = { length };
    boost::array<char, 1> ready = {};

    boost::system::error_code error;
    socket.write_some(boost::asio::buffer(size), error);

    socket.read_some(boost::asio::buffer(ready), error);
    socket.write_some(boost::asio::buffer(buf.data(), buf.size()), error);

    socket.write_some(boost::asio::buffer(ready), error);
    socket.read_some( boost::asio::buffer(size), error );
    buf.clear();
    buf.resize(size[0]);

    socket.write_some(boost::asio::buffer(ready), error);
    boost::asio::read(socket, boost::asio::buffer(buf.data(), buf.size()), error);
    //socket.read_some(boost::asio::buffer(buf.data(), buf.size()), error);

    std::ofstream os( outputFileName, std::ios_base::out | std::ios_base::binary );
    if ( !os.is_open() )
        throw std::runtime_error( "unable to open input file" );

    os.write(buf.data(), buf.size());

    if ( !os )
        throw std::runtime_error( "unable to write the file" );
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
