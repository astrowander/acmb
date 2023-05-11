#include "server.h"
#include "tools.h"

#include "./../Core/bitmap.h"
#include "./../Codecs/PPM/ppmencoder.h"
#include "./../Transforms/ResizeTransform.h"

#include <boost/array.hpp>
#include <thread>

using boost::asio::ip::tcp;

ACMB_SERVER_NAMESPACE_BEGIN

void Server::ListenClientPort(uint16_t port)
{
    tcp::acceptor acceptor( context_, tcp::endpoint(tcp::v4(), port ) );
    tcp::socket socket(context_);
    acceptor.accept(socket);    

    std::string data = ReceiveData(socket);

    auto pStream = std::make_shared<std::istringstream>( data );
    auto pBitmap = IBitmap::Create( pStream );
    pBitmap = ResizeTransform::Resize(pBitmap, { pBitmap->GetWidth() / 2, pBitmap->GetHeight() / 2 } );

    std::shared_ptr<PpmEncoder> pEncoder = std::make_shared<PpmEncoder>( PpmMode::Binary );
    auto pOutputStream = std::make_shared<std::ostringstream>();
    pEncoder->Attach( pOutputStream );
    pEncoder->WriteBitmap(pBitmap);

    SendData(socket, pOutputStream->str());
}

void Server::ListenHelloPort()
{
    tcp::acceptor acceptor( context_, tcp::endpoint(tcp::v4(), cHelloPort ) );
    for (;;)
    {
      tcp::socket socket(context_);
      acceptor.accept(socket);

      int command = ReceiveSingleObject<int>( socket );
      int portNumber = ReceiveSingleObject<int>( socket );
      int answer = -1;
      switch ( command )
      {
      case 1:
          if ( !_activeConnections.all() )
          {
              for ( int i = 0; i < cPortPoolSize; ++i )
              {
                  if ( !_activeConnections.test(i) )
                  {
                      _activeConnections.set(i, true);
                      answer = cHelloPort + i + 1;
                      std::thread thread( [this, answer]{this->ListenClientPort(answer);});
                      thread.detach();
                      break;
                  }
              }
          }

          break;
      case 2:
      {
          const size_t pos = portNumber - cHelloPort - 1;
          if (_activeConnections.test( pos ))
          {
              _activeConnections.set(pos, false);
              answer = 0;
          }       
          break;
      }
      default:
          break;
      }

      SendSingleObject<int>( socket, answer );
    }
}

void Server::Launch()
{
    Server server;
    server.ListenHelloPort();
}

ACMB_SERVER_NAMESPACE_END
