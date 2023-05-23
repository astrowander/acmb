#pragma once
#include "./../Core/macros.h"
#include <boost/asio.hpp>
#include <bitset>
ACMB_SERVER_NAMESPACE_BEGIN

static constexpr int cHelloPort = 49999;
static constexpr int cPortPoolSize = 16;

class Server
{
    std::bitset<cPortPoolSize> _activeConnections;
    boost::asio::io_context context_;

    Server() = default;
    void ListenHelloPort();
    void ListenClientPort( uint16_t port );
    void WorkWithClient( boost::asio::ip::tcp::socket& socket );
public:
    static void Launch();

};

ACMB_SERVER_NAMESPACE_END
