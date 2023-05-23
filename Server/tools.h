#pragma once
#include "./../Core/macros.h"
#include <boost/asio.hpp>
#include <boost/array.hpp>
#include <string>

ACMB_SERVER_NAMESPACE_BEGIN

void SendOKCode( boost::asio::ip::tcp::socket& socket );
void SendError( boost::asio::ip::tcp::socket& socket, const std::string& errMsg );

template <typename T>
void SendSingleObject( boost::asio::ip::tcp::socket& socket, T data )
{
    SendOKCode( socket );
    boost::system::error_code error;
    const boost::array<T, 1> arr = { std::move( data ) };
    boost::asio::write( socket, boost::asio::buffer( arr ), error );
}

template <typename T>
T ReceiveSingleObject( boost::asio::ip::tcp::socket& socket )
{
    SendOKCode( socket );
    boost::array<T, 1> arr = {};
    boost::system::error_code error;
    boost::asio::read( socket, boost::asio::buffer(arr), error );
    return arr[0];
}

std::string ReceiveData( boost::asio::ip::tcp::socket& socket );
void SendData( boost::asio::ip::tcp::socket& socket, const std::string& data );

ACMB_SERVER_NAMESPACE_END
