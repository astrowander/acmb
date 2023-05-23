#include "tools.h"
#include <fstream>

using boost::asio::ip::tcp;

ACMB_CLIENT_NAMESPACE_BEGIN

void CheckServerError( boost::asio::ip::tcp::socket& socket )
{
    boost::system::error_code error;
    
    boost::array<size_t, 1> size = {};    
    boost::asio::read( socket, boost::asio::buffer( size ), error );
    if ( size[0] == 0 )
        return;

    std::string res;
    res.resize( size[0] );
    boost::asio::read( socket, boost::asio::buffer( res.data(), res.size() ), error );

    throw std::runtime_error( res );
}

void UploadData( tcp::socket& socket, std::string data )
{
    boost::system::error_code error;
    UploadSingleObject<size_t>(socket, data.size() );
    boost::asio::write( socket, boost::asio::buffer(data.data(), data.size()), error );
}

std::string DownloadData( tcp::socket& socket )
{
    boost::system::error_code error;
    const size_t size = DownloadSingleObject<size_t>( socket );
    std::string res;
    res.resize( size );
    boost::asio::read(socket, boost::asio::buffer( res.data(), res.size() ), error);
    return res;
}

void UploadFile( tcp::socket& socket, const std::string& fileName )
{
    std::ifstream is( fileName, std::ios_base::in | std::ios_base::binary );
    if ( !is.is_open() )
        throw std::runtime_error( "unable to open input file" );

    std::string buf;
    is.seekg( 0, is.end );
    const size_t length = is.tellg();
    is.seekg( 0, is.beg );

    buf.resize( length );
    is.read( buf.data(), length );

    if ( !is )
        throw std::runtime_error( "unable to read the file" );

    UploadData( socket, std::move( buf ) );
}

void DownloadFile( tcp::socket& socket, const std::string& fileName )
{
    const auto buf = DownloadData( socket );

    std::ofstream os( fileName, std::ios_base::out | std::ios_base::binary );
    if ( !os.is_open() )
        throw std::runtime_error( "unable to open input file" );

    os.write(buf.data(), buf.size());

    if ( !os )
        throw std::runtime_error( "unable to write the file" );
}

ACMB_CLIENT_NAMESPACE_END
