#pragma once
#include "./../Core/macros.h"
#include <boost/asio.hpp>
#include <boost/array.hpp>
#include <vector>

ACMB_CLIENT_NAMESPACE_BEGIN

template <typename T>
void UploadSingleObject( boost::asio::ip::tcp::socket& socket, T data )
{
    boost::system::error_code error;
    const boost::array<T, 1> arr = { std::move( data ) };
    boost::asio::write( socket, boost::asio::buffer( arr ), error );
}

template <typename T>
T DownloadSingleObject( boost::asio::ip::tcp::socket& socket )
{
    boost::array<T, 1> arr = {};
    boost::system::error_code error;
    boost::asio::read( socket, boost::asio::buffer(arr), error );
    return arr[0];
}

void UploadData( boost::asio::ip::tcp::socket& socket, std::string data );
std::string DownloadData( boost::asio::ip::tcp::socket& socket );

void UploadFile( boost::asio::ip::tcp::socket& socket, const std::string& fileName );
void DownloadFile( boost::asio::ip::tcp::socket& socket, const std::string& fileName );

ACMB_CLIENT_NAMESPACE_END
