#include "server.h"

#include "./../Core/bitmap.h"
#include "./../Codecs/PPM/ppmencoder.h"
#include "./../Transforms/ResizeTransform.h"

#include <boost/array.hpp>
#include <thread>
#include <vector>
using boost::asio::ip::tcp;

ACMB_SERVER_NAMESPACE_BEGIN

struct membuf : std::streambuf
{
    membuf(char* begin, char* end)
    {
        this->setg(begin, begin, end);
    }
};

void Server::ListenClientPort(uint16_t port)
{
    tcp::acceptor acceptor( context_, tcp::endpoint(tcp::v4(), port ) );

    tcp::socket socket(context_);
    acceptor.accept(socket);

    boost::array<char, 1> ready = {};

    boost::system::error_code ignored_error;
    boost::array<size_t, 1> size = {};
    boost::asio::read(socket, boost::asio::buffer( size ), ignored_error);
    boost::asio::write(socket, boost::asio::buffer(ready), ignored_error);

    std::string data;
    data.resize(size[0]);
    boost::asio::read(socket, boost::asio::buffer( data.data(), data.size() ), ignored_error);

    auto pStream = std::make_shared<std::istringstream>( data );
    auto pBitmap = IBitmap::Create( pStream );
    pBitmap = ResizeTransform::Resize(pBitmap, { pBitmap->GetWidth() / 2, pBitmap->GetHeight() / 2 } );

    data.clear();
//    data.resize( pBitmap->GetByteSize() );

    std::shared_ptr<PpmEncoder> pEncoder = std::make_shared<PpmEncoder>( PpmMode::Binary );
    auto pOutputStream = std::make_shared<std::ostringstream>();
    pEncoder->Attach( pOutputStream );
    pEncoder->WriteBitmap(pBitmap);

    const auto str = pOutputStream->str();
    const char ch = str[0x10000];
    size[0] = pBitmap->GetByteSize();
    boost::asio::read(socket, boost::asio::buffer( ready ), ignored_error);
    boost::asio::write(socket, boost::asio::buffer( size ));
    boost::asio::read(socket, boost::asio::buffer( ready ), ignored_error);
    boost::asio::write(socket, boost::asio::buffer( str.data(), str.size() ), ignored_error);
}

void Server::ListenHelloPort()
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
                      std::thread thread( [this, answer]{this->ListenClientPort(answer[0]);});
                      thread.detach();
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
    server.ListenHelloPort();
}

ACMB_SERVER_NAMESPACE_END
