#include "tools.h"
using boost::asio::ip::tcp;

ACMB_SERVER_NAMESPACE_BEGIN

void SendOKCode( tcp::socket& socket )
{
    boost::system::error_code ignored_error;
    boost::array<size_t, 1> size = { 0 };
    boost::asio::write( socket, boost::asio::buffer( size ), ignored_error );
}

void SendError( boost::asio::ip::tcp::socket& socket, const std::string& errMsg )
{
    boost::system::error_code ignored_error;
    boost::array<size_t, 1> size = { errMsg.size() };
    boost::asio::write( socket, boost::asio::buffer( size ) );
    boost::asio::write( socket, boost::asio::buffer( errMsg.data(), errMsg.size() ), ignored_error );
}

std::string ReceiveData( tcp::socket& socket )
{
    SendOKCode( socket );
    boost::system::error_code ignored_error;
    boost::array<size_t, 1> size = {};
    boost::asio::read(socket, boost::asio::buffer( size ), ignored_error);

    std::string data;
    data.resize(size[0]);
    boost::asio::read(socket, boost::asio::buffer( data.data(), data.size() ), ignored_error);
    return data;
}

void SendData( tcp::socket& socket, const std::string& data )
{
    SendOKCode( socket );
    boost::system::error_code ignored_error;
    boost::array<size_t, 1> size = { data.size() };
    boost::asio::write(socket, boost::asio::buffer( size ));
    boost::asio::write(socket, boost::asio::buffer( data.data(), data.size() ), ignored_error);
}

ACMB_SERVER_NAMESPACE_END
