#pragma once
#include "./../Core/macros.h"
#include <boost/asio.hpp>

ACMB_CLIENT_NAMESPACE_BEGIN

static constexpr int cHelloPort = 49999;

class Client
{
    boost::asio::io_context context_;
    std::string serverAddress_;
    int portNumber_ = -1;

public:
    Client(const std::string& serverAddress);
    void Connect();
    void Disconnect();
    int portNumber() { return portNumber_; }
};

ACMB_CLIENT_NAMESPACE_END
