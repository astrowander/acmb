#include "client.h"
#include "tools.h"
#include "enums.h"
#include "CliParser.h"

#include "./../Core/enums.h"
#include <filesystem>
#include <set>
#include <algorithm>
#include <unordered_map>

using boost::asio::ip::tcp;

ACMB_CLIENT_NAMESPACE_BEGIN

static const std::unordered_map<std::string, PixelFormat> stringToPixelFormat =
{
    {"gray8", PixelFormat::Gray8},
    {"gray16", PixelFormat::Gray16},
    {"rgb24", PixelFormat::RGB24},
    {"rgb48", PixelFormat::RGB48}
};

std::string IntToString( size_t num, size_t minDigitCount )
{
    auto res = std::to_string( num );
    if ( res.size() < minDigitCount )
    {
        res.insert( 0, std::string( minDigitCount - res.size(), '0' ) );
    }

    return res;
}

std::string ToLower(const std::string& str)
{
    std::string res;
    res.resize( str.size() );
    std::transform( str.begin(), str.end(), res.begin(), [](char ch) {return std::tolower(ch);} );
    return res;
}

std::vector<std::string> GetFileNamesFromDir( const std::string& path )
{
    if ( !std::filesystem::is_directory( path ) )
        return {};

    std::set<std::string> sortedNames;
    for ( const auto& entry : std::filesystem::directory_iterator( path) )
    {
        if ( std::filesystem::is_directory( entry ) )
            continue;

        sortedNames.insert( entry.path().string() );
    }

    return std::vector<std::string>( sortedNames.begin(), sortedNames.end() );
}

std::vector<std::string> GetFileNamesFromMask( const std::string& mask )
{
    size_t start = 0;
    std::vector<std::string> res;

    while ( start < mask.size() )
    {
        auto end = mask.find_first_of( ';', start );
        std::string fileName = mask.substr( start, end - start );
        size_t separatorPos = fileName.find_first_of( '#' );

        if ( std::filesystem::exists( fileName ) )
        {
            if ( std::filesystem::is_directory( fileName ) )
            {
                auto fileNames = GetFileNamesFromDir( fileName );
                res.insert( res.end(), fileNames.begin(), fileNames.end() );
            }
            else
            {
                res.push_back( fileName );
            }
        }
        else if ( separatorPos != std::string::npos )
        {
            size_t pointPos = fileName.find_first_of( '.', separatorPos );
            auto varDigitCount = pointPos - separatorPos - 1;
            if ( varDigitCount != 0 )
            {

                int minNum = std::stoi( fileName.substr( separatorPos - varDigitCount, varDigitCount ) );
                int maxNum = std::stoi( fileName.substr( separatorPos + 1, varDigitCount ) );

                for ( int j = minNum; j <= maxNum; ++j )
                {
                    auto tempName = fileName.substr( 0, separatorPos - varDigitCount ) + IntToString( j, varDigitCount ) + fileName.substr( pointPos );
                    if ( std::filesystem::exists( tempName ) )
                        res.push_back( tempName );
                }
            }
        }

        if ( end == std::string::npos )
            break;

        start = end + 1;
    }

    return res;
}

Client::Client(const std::string& serverAddress)
:serverAddress_(serverAddress)
{
}

void Client::Connect()
{
    const auto ipAddr = boost::asio::ip::address::from_string( serverAddress_ );
    boost::array< tcp::endpoint, 1> endpoints = { tcp::endpoint( ipAddr, cHelloPort ) };

    tcp::socket socket_( context_ );
    boost::asio::connect(socket_, endpoints );

    UploadSingleObject<int>(socket_, 1);
    UploadSingleObject<int>(socket_, 0);
    portNumber_ = DownloadSingleObject<int>( socket_ );
    if ( portNumber_ == -1)
        throw std::runtime_error("Unable to connect");
}

void Client::Process( const std::vector<std::string>& args )
{
    auto kvs = Parse( args );

    if ( kvs.size() < 2 ||
         kvs.front().key != "--input" || kvs.front().values.empty() ||
         kvs.back().key != "--output" || kvs.back().values.empty() )
    {
        throw std::runtime_error( "Invalid command line" );
    }

    const auto inputFiles = GetFileNamesFromMask( kvs.front().values[0] );
    if ( inputFiles.empty() )
        throw std::runtime_error( "No input files" );

    //auto pFirstDecoder = ImageDecoder::Create( inputFiles[0] );
    //const auto inputPixelFormat = pFirstDecoder->GetPixelFormat();
    const auto outputPath = kvs.back().values[0];

    bool isStackerFound = false;

    const auto ipAddr = boost::asio::ip::address::from_string( serverAddress_ );
    const boost::array<tcp::endpoint, 1> endpoints = { tcp::endpoint( ipAddr, portNumber_ ) };
    tcp::socket socket( context_ );
    boost::asio::connect(socket, endpoints );    

    UploadFile( socket, inputFiles[0] );
    std::cout << "1 of " << inputFiles.size() << " is uploaded" << std::endl;

    std::string inputExtension = inputFiles[0].substr( inputFiles[0].find_last_of('.') );
    UploadData( socket, std::move(inputExtension ) );

    UploadSingleObject( socket, kvs.size() - 2 );
    for (size_t i = 1; i < kvs.size() - 1; ++i )
    {
        const auto& kv = kvs[i];

        if ( kv.key == "--desiredFormat" )
        {
            if ( kv.values.empty() )
                throw std::runtime_error("Wrong command args");

            const auto it = stringToPixelFormat.find( ToLower( kv.values[0] ) );
            if ( it == stringToPixelFormat.end() )
                throw std::runtime_error("Wrong command args");

            UploadSingleObject( socket, CommandCode::SetDesiredFormat );
            UploadSingleObject( socket, it->second );
        }
        else if ( kv.key == "--binning" )
        {
            if ( kv.values.empty() )
                throw std::runtime_error("Wrong command args");

            if ( kv.values.size() != 2 )
                throw std::runtime_error( "--binning requires exactly two arguments" );

            const uint32_t width = std::stoi( kv.values[0] );
            const uint32_t height = std::stoi( kv.values[1] );
            if ( width <= 0 || height <= 0 )
                throw std::runtime_error( "--binning requires strictly positive arguments" );

            UploadSingleObject( socket, CommandCode::Binning );
            UploadSingleObject( socket, width );
            UploadSingleObject( socket, height );
        }
        else if ( kv.key == "--convert" )
        {
            if ( kv.values.empty() )
                throw std::runtime_error("Wrong command args");

            const auto it = stringToPixelFormat.find( ToLower( kv.values[0] ) );
            if ( it == stringToPixelFormat.end() )
                throw std::runtime_error("Wrong command args");

            UploadSingleObject( socket, CommandCode::Convert );
            UploadSingleObject( socket, it->second );
        }
        else if ( kv.key == "--subtract" )
        {
            if ( kv.values.size() != 1 )
               throw std::runtime_error( "--subtract requires exactly one argument" );

            UploadSingleObject( socket, CommandCode::Subtract );
            UploadFile( socket, kv.values[0] );
        }
        else if ( kv.key == "--divide" )
        {
            if ( kv.values.size() != 1 && kv.values.size() != 2 )
               throw std::runtime_error( "--divide requires one or two arguments" );

            float intensity = 100.0f;
            if ( kv.values.size() == 2 )
                intensity = std::stof( kv.values[1] );

            UploadSingleObject( socket, CommandCode::Divide );
            UploadSingleObject( socket, intensity );
            UploadFile( socket, kv.values[0] );
        }
        else if ( kv.key == "--autowb" )
        {
            if ( !kv.values.empty() )
               throw std::runtime_error( "--autowb requires zero arguments" );

            UploadSingleObject( socket, CommandCode::AutoWB );
        }
        else if ( kv.key == "--deaberrate" )
        {
            if ( !kv.values.empty() )
               throw std::runtime_error( "--deaberrate requires zero arguments" );

            UploadSingleObject( socket, CommandCode::Deaberrate );
        }
        else if (kv.key == "--removehalo")
        {
            if ( kv.values.size() > 1 )
                throw std::runtime_error( "--removehalo requires zero or one arguments" );

            float intensity = 100.0f;
            if ( kv.values.size() == 1 )
                intensity = std::stof( kv.values[0] );

            UploadSingleObject( socket, CommandCode::RemoveHalo );
            UploadSingleObject( socket, intensity );
        }
        else if ( kv.key == "--resize" )
        {
            if ( kv.values.size() != 2 )
               throw std::runtime_error( "--resize requires exactly two arguments" );

            const auto width = uint32_t( std::stoi( kv.values[0] ) );
            const auto height = uint32_t( std::stoi( kv.values[1] ) );

            UploadSingleObject( socket, CommandCode::Resize );
            UploadSingleObject( socket, width );
            UploadSingleObject( socket, height );
        }
        else if ( kv.key == "--crop" )
        {
            if ( kv.values.size() != 4 )
               throw std::runtime_error( "--crop requires exactly 4 arguments" );

            const auto x = uint32_t( std::stoi( kv.values[0] ) );
            const auto y = uint32_t( std::stoi( kv.values[1] ) );
            const auto width = uint32_t( std::stoi( kv.values[2] ) );
            const auto height = uint32_t( std::stoi( kv.values[3] ) );

            UploadSingleObject( socket, CommandCode::Resize );
            UploadSingleObject( socket, x );
            UploadSingleObject( socket, y );
            UploadSingleObject( socket, width );
            UploadSingleObject( socket, height );
        }
        else if ( kv.key == "--debayer" )
        {
            if ( !kv.values.empty() )
               throw std::runtime_error( "--debayer requires zero arguments" );

            UploadSingleObject( socket, CommandCode::Debayer );
        }
        else if ( kv.key == "--stack" )
        {
            if ( isStackerFound )
                throw std::runtime_error( "only one --stack is allowed" );

            isStackerFound = true;

            bool enableCudaIfAvailable = false;
            if ( kv.values.size() >= 2 && kv.values[1] == "usecuda" )
                enableCudaIfAvailable = true;

            StackMode mode = StackMode::Light;
            if ( !kv.values.empty() )
            {
                const auto strMode = ToLower( kv.values[0] );
                if ( strMode == "dark" || strMode == "flat" )
                    mode = StackMode::DarkOrFlat;
                else if ( strMode == "noalign" )
                    mode = StackMode::LightNoAlign;
            }

            UploadSingleObject( socket, CommandCode::Stack );
            UploadSingleObject( socket, mode );
            UploadSingleObject( socket, enableCudaIfAvailable );
        }
    }

    std::string outputExtension;

    if (!isStackerFound && kvs.back().values.size() >= 2 )
    {
        outputExtension = kvs.back().values[1];
    }
    else
    {
        const auto dotPos = outputPath.find_last_of( '.' );
        outputExtension = outputPath.substr( dotPos );
    }

    UploadData(socket, outputExtension );
    UploadSingleObject( socket, inputFiles.size() );
    if ( !isStackerFound )
    {
        const auto lastSlashPos = inputFiles[0].find_last_of("/\\");
        const auto dotPos = inputFiles[0].find_last_of( '.' );
        DownloadFile(socket, outputPath + "/" + inputFiles[0].substr(lastSlashPos + 1, dotPos - lastSlashPos - 1 ) + outputExtension );
    }

    for ( size_t i = 1; i < inputFiles.size(); ++i )
    {
        const auto& inputFile = inputFiles[i];
        inputExtension = inputFile.substr( inputFile.find_last_of('.') );

        UploadFile( socket, inputFile );
        std::cout << i + 1 << " of " << inputFiles.size() << " are uploaded" << std::endl;
        UploadData( socket, std::move( inputExtension ) );
        if (!isStackerFound)
        {
            const auto lastSlashPos = inputFile.find_last_of("/\\");
            const auto dotPos = inputFile.find_last_of( '.' );
            DownloadFile(socket, outputPath + "/" + inputFile.substr(lastSlashPos + 1, dotPos - lastSlashPos - 1 ) + outputExtension );
        }
    }

    if ( isStackerFound )
        DownloadFile( socket, outputPath );

    std::cout << "Result is downloaded" << std::endl;
}

void Client::Disconnect()
{
    const auto ipAddr = boost::asio::ip::address::from_string( serverAddress_ );
    boost::array< tcp::endpoint, 1> endpoints = { tcp::endpoint( ipAddr, cHelloPort ) };

    tcp::socket socket_( context_ );
    boost::asio::connect(socket_, endpoints );
    UploadSingleObject<int>( socket_, 2 );
    UploadSingleObject<int>( socket_, portNumber_ );
    const auto answer = DownloadSingleObject<int>( socket_ );
    if ( answer == -1)
        throw std::runtime_error("Unable to disconnect");

    portNumber_ = -1;
}

ACMB_CLIENT_NAMESPACE_END
