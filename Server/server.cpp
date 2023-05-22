#include "server.h"
#include "./../Client/enums.h"
#include "tools.h"

#include "./../Core/bitmap.h"
#include "./../Codecs/PPM/ppmencoder.h"
#include "./../Codecs/Tiff/TiffEncoder.h"
#include "./../Codecs/JPEG/JpegEncoder.h"

#include "./../Codecs/PPM/ppmdecoder.h"
#include "./../Codecs/Raw/RawDecoder.h"
#include "./../Codecs/Tiff/TiffDecoder.h"

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
#include "./../Cuda/CudaStacker.h"

#include "./../Cuda/CudaInfo.h"

#include <boost/array.hpp>
#include <thread>
#include <sstream>

using boost::asio::ip::tcp;

ACMB_SERVER_NAMESPACE_BEGIN

std::string ToLower(const std::string& str)
{
    std::string res;
    res.resize( str.size() );
    std::transform( str.begin(), str.end(), res.begin(), [](char ch) {return std::tolower(ch);} );
    return res;
}

IBitmapPtr ReadBitmap( std::shared_ptr<std::istringstream> pStream, const std::string& extension )
{
    std::shared_ptr<ImageDecoder> pDecoder;
    if ( TiffDecoder::GetExtensions().contains(extension) )
        pDecoder = std::make_shared<TiffDecoder>();
    else if ( PpmDecoder::GetExtensions().contains(extension) )
        pDecoder = std::make_shared<PpmDecoder>();
    else if ( RawDecoder::GetExtensions().contains(extension) )
        pDecoder = std::make_shared<RawDecoder>();
    else
        throw std::runtime_error( "Unsupported extension" );

    pDecoder->Attach( pStream );
    auto res = pDecoder->ReadBitmap();
    return res;
}

void WriteBitmap( tcp::socket& socket, IBitmapPtr pBitmap, const std::string& extension )
{
    std::shared_ptr<ImageEncoder> pEncoder;
    if ( TiffEncoder::GetExtensions().contains(extension) )
        pEncoder = std::make_shared<TiffEncoder>();
    else if ( PpmEncoder::GetExtensions().contains(extension) )
        pEncoder = std::make_shared<PpmEncoder>( PpmMode::Binary );
    else if ( JpegEncoder::GetExtensions().contains(extension) )
        pEncoder = std::make_shared<JpegEncoder>();

    auto pOutputStream = std::make_shared<std::ostringstream>();
    pEncoder->Attach( pOutputStream );
    pEncoder->WriteBitmap( pBitmap );
    SendData(socket, pOutputStream->str());
}

void Server::ListenClientPort(uint16_t port)
{
    tcp::acceptor acceptor( context_, tcp::endpoint(tcp::v4(), port ) );
    tcp::socket socket(context_);
    acceptor.accept(socket);

    std::string str = ReceiveData(socket);
    std::string inputExtension = ToLower( ReceiveData(socket) );
    auto pInStream = std::make_shared<std::istringstream>( str );    

    const size_t commandCount = ReceiveSingleObject<size_t>( socket );
    std::vector<CommandCode> commands( commandCount );

    std::shared_ptr<BaseStacker> pStacker;
    Pipeline beforeStacker( ReadBitmap( pInStream, inputExtension ) );
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
            const auto pixelFormat = activePipeline->GetFinalParams()->GetPixelFormat();
            activePipeline->AddTransform<BitmapSubtractor>( IBitmap::Create( pStream, pixelFormat ) );
            break;
        }
        case CommandCode::Divide:
        {
            const float intensity = ReceiveSingleObject<float>( socket );
            std::string str = ReceiveData(socket);
            auto pStream = std::make_shared<std::istringstream>( str );
            const auto pixelFormat = activePipeline->GetFinalParams()->GetPixelFormat();
            auto pBitmapToDivide = IBitmap::Create( pStream, pixelFormat );
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
            if ( cuda::isCudaAvailable() )
                pStacker = std::make_shared<cuda::Stacker>( *finalParams, stackMode );
            else
                pStacker = std::make_shared<Stacker>( *finalParams, stackMode );

            pStackedBitmap = IBitmap::Create(finalParams->GetWidth(), finalParams->GetHeight(), finalParams->GetPixelFormat() );
            afterStacker = Pipeline { pStackedBitmap };
            activePipeline = &afterStacker;
        }
        default:
            break;
        }
    }

    const std::string outputExtension = ToLower( ReceiveData( socket ) );
    const size_t fileCount = ReceiveSingleObject<size_t>( socket );

    if ( pStacker)
        pStacker->AddBitmap( beforeStacker );
    else
        WriteBitmap( socket, beforeStacker.RunAndGetBitmap(), outputExtension );

    for ( size_t i = 1; i < fileCount; ++i )
    {
        std::string str = ReceiveData( socket );
        inputExtension = ToLower( ReceiveData( socket ) );

        auto pStream = std::make_shared<std::istringstream>( str );
        if ( !ImageDecoder::GetAllExtensions().contains( inputExtension ) )
            continue;

        beforeStacker.ReplaceFirstElement(ReadBitmap(pStream, inputExtension));
        if ( pStacker)
            pStacker->AddBitmap( beforeStacker );
        else
            WriteBitmap( socket, beforeStacker.RunAndGetBitmap(), outputExtension );
    }

    if ( pStacker )
    {
        pStackedBitmap = pStacker->GetResult();
        afterStacker.ReplaceFirstElement( pStackedBitmap );
        WriteBitmap(socket, afterStacker.RunAndGetBitmap(), outputExtension );
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
