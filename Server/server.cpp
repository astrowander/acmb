#include "server.h"
#include "./../Client/enums.h"
#include "tools.h"

#include "./../Core/bitmap.h"
#include "./../Codecs/PPM/ppmencoder.h"
#include "./../Codecs/Tiff/TiffEncoder.h"
#include "./../Codecs/JPEG/JpegEncoder.h"

#include "./../Transforms/binningtransform.h"
#include "./../Transforms/converter.h"
#include "./../Transforms/BitmapSubtractor.h"
#include "./../Transforms/BitmapDivisor.h"
#include "./../Transforms/ChannelEqualizer.h"
#include "./../Transforms/deaberratetransform.h"
#include "./../Transforms/DebayerTransform.h"
#include "./../Transforms/ResizeTransform.h"
#include "./../Transforms/CropTransform.h"
#include "./../Transforms/HaloRemovalTransform.h"

#include "./../Registrator/stacker.h"

#include <boost/array.hpp>
#include <thread>
#include <sstream>

using boost::asio::ip::tcp;

ACMB_SERVER_NAMESPACE_BEGIN

std::shared_ptr<ImageEncoder> CreateEncoder( ExtensionCode exCode )
{
    switch ( exCode )
    {
    case ExtensionCode::Ppm:
        return std::make_shared<PpmEncoder>( PpmMode::Binary );
    case ExtensionCode::Tiff:
        return std::make_shared<TiffEncoder>();
    case ExtensionCode::Jpeg:
        return std::make_shared<JpegEncoder>();
    }
}

void Server::ListenClientPort(uint16_t port)
{
    tcp::acceptor acceptor( context_, tcp::endpoint(tcp::v4(), port ) );
    tcp::socket socket(context_);
    acceptor.accept(socket);

    std::string str = ReceiveData(socket);
    auto pInStream = std::make_shared<std::istringstream>( str );

    const size_t commandCount = ReceiveSingleObject<size_t>( socket );
    std::vector<CommandCode> commands( commandCount );

    std::shared_ptr<Stacker> pStacker;
    Pipeline beforeStacker( IBitmap::Create( pInStream ) );
    Pipeline afterStacker;
    Pipeline* activePipeline = &beforeStacker;

    IBitmapPtr pStackedBitmap;

    for ( size_t i = 0; i < commandCount; ++i )
    {
        IPipelineElementPtr pElement;
        const auto commandCode = ReceiveSingleObject<CommandCode>( socket );
        switch ( commandCode )
        {
        case CommandCode::Binning:
        {
            const uint32_t width = ReceiveSingleObject<int>( socket );
            const uint32_t height = ReceiveSingleObject<int>( socket );
            Size bin {width, height};
            activePipeline->AddTransform<BinningTransform>( bin );
            break;
        }
        case CommandCode::SetDesiredFormat:
        case CommandCode::Convert:
        {
            activePipeline->AddTransform<Converter>(ReceiveSingleObject<PixelFormat>( socket ) );
            break;
        }
        case CommandCode::Subtract:
        {
            std::string str = ReceiveData(socket);
            auto pStream = std::make_shared<std::istringstream>( str );
            activePipeline->AddTransform<BitmapSubtractor>( IBitmap::Create( pStream ) );
            break;
        }
        case CommandCode::Divide:
        {
            const float intensity = ReceiveSingleObject<float>( socket );
            std::string str = ReceiveData(socket);
            auto pStream = std::make_shared<std::istringstream>( str );
            auto pBitmapToDivide = IBitmap::Create( pStream );
            activePipeline->AddTransform<BitmapDivisor>(BitmapDivisor::Settings{ pBitmapToDivide, intensity } );
            break;
        }
        case CommandCode::AutoWB:
        {
            activePipeline->AddTransform<AutoChannelEqualizer>();
            break;
        }
        case CommandCode::Deaberrate:
        {
            activePipeline->AddTransform<DeaberrateTransform>();
            break;
        }
        case CommandCode::Resize:
        {
            const uint32_t width = ReceiveSingleObject<uint32_t>( socket );
            const uint32_t height = ReceiveSingleObject<uint32_t>( socket );
            activePipeline->AddTransform<ResizeTransform>( Size{width, height} );
            break;
        }
        case CommandCode::Crop:
        {
            const int x = ReceiveSingleObject<uint32_t>( socket );
            const int y = ReceiveSingleObject<uint32_t>( socket );
            const int width = ReceiveSingleObject<uint32_t>( socket );
            const int height = ReceiveSingleObject<uint32_t>( socket );
            activePipeline->AddTransform<CropTransform>( Rect{x, y, width, height} );
            break;
        }
        case CommandCode::Debayer:
        {
            activePipeline->AddTransform<DebayerTransform>();
            break;
        }
        case CommandCode::RemoveHalo:
        {
            activePipeline->AddTransform<AutoHaloRemoval>(ReceiveSingleObject<float>( socket ));
            break;
        }
        case CommandCode::Stack:
        {
            const StackMode stackMode = ReceiveSingleObject<StackMode>(socket );
            const auto enableCudaIfAvailable = ReceiveSingleObject<bool>( socket );
            const auto finalParams = beforeStacker.GetFinalParams();
            pStacker = std::make_shared<Stacker>( *finalParams, stackMode );
            pStackedBitmap = IBitmap::Create(finalParams->GetWidth(), finalParams->GetHeight(), finalParams->GetPixelFormat() );
            afterStacker = Pipeline { pStackedBitmap };
            activePipeline = &afterStacker;
        }
        default:
            break;
        }
    }

    const ExtensionCode exCode = ReceiveSingleObject<ExtensionCode>( socket );
    const size_t fileCount = ReceiveSingleObject<size_t>( socket );
    for ( size_t i = 0; i < fileCount; ++i )
    {
        std::string str = ReceiveData(socket);
        auto pStream = std::make_shared<std::istringstream>( str );
        beforeStacker.ReplaceFirstElement(IBitmap::Create(pStream));
        if ( pStacker )
        {
            pStacker->AddBitmap( beforeStacker );
            continue;
        }

        auto pEncoder = CreateEncoder( exCode );
        auto pOutputStream = std::make_shared<std::ostringstream>();
        pEncoder->Attach( pOutputStream );
        pEncoder->WriteBitmap( beforeStacker.RunAndGetBitmap() );
        SendData(socket, pOutputStream->str());
    }

    if ( pStacker )
    {
        pStackedBitmap = pStacker->GetResult();
        afterStacker.ReplaceFirstElement( pStackedBitmap );
        auto pEncoder = CreateEncoder( exCode );
        auto pOutputStream = std::make_shared<std::ostringstream>();
        pEncoder->Attach( pOutputStream );
        pEncoder->WriteBitmap( afterStacker.RunAndGetBitmap() );
        SendData(socket, pOutputStream->str());
    }
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
