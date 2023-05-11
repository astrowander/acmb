#include "tools.h"
using boost::asio::ip::tcp;

ACMB_SERVER_NAMESPACE_BEGIN

std::string ReceiveData( tcp::socket& socket )
{
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
    boost::system::error_code ignored_error;
    boost::array<size_t, 1> size = { data.size() };
    boost::asio::write(socket, boost::asio::buffer( size ));
    boost::asio::write(socket, boost::asio::buffer( data.data(), data.size() ), ignored_error);
}

ACMB_SERVER_NAMESPACE_END
