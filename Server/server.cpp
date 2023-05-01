#include "server.h"
#include <boost/array.hpp>
using boost::asio::ip::tcp;

ACMB_SERVER_NAMESPACE_BEGIN

void Server::Listen()
{
    tcp::acceptor acceptor( context_, tcp::endpoint(tcp::v4(), cHelloPort ) );
    for (;;)
    {
      tcp::socket socket(context_);
      acceptor.accept(socket);

      boost::system::error_code ignored_error;
      boost::array<int, 2> command = { 0 };
      boost::array<int, 1> answer = { -1 };

      boost::asio::read(socket, boost::asio::buffer( command ), ignored_error);

      switch ( command[0] )
      {
      case 1:
          if ( !_activeConnections.all() )
          {
              for ( int i = 0; i < cPortPoolSize; ++i )
              {
                  if ( !_activeConnections.test(i) )
                  {
                      _activeConnections.set(i, true);
                      answer[0] = cHelloPort + i + 1;
                      break;
                  }
              }
          }
          boost::asio::write(socket, boost::asio::buffer( answer ), ignored_error);
          break;
      case 2:
      {
          const size_t pos = command[1] - cHelloPort - 1;
          if (_activeConnections.test( pos ))
          {
              _activeConnections.set(pos, false);
              answer[0] = 0;
          }
          boost::asio::write(socket, boost::asio::buffer( answer ), ignored_error);
          break;
      }
      default:
          break;
      }
    }
}

void Server::Launch()
{
    Server server;
    server.Listen();
}

ACMB_SERVER_NAMESPACE_END
