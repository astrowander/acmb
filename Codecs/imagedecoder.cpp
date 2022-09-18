#include "imagedecoder.h"

#include "PPM/ppmdecoder.h"
#include "Raw/RawDecoder.h"
#include "Tiff/TiffDecoder.h"

#include <fstream>
#include <filesystem>
#include <set>
#include <algorithm>

ACMB_NAMESPACE_BEGIN
std::string IntToString( size_t num, size_t minDigitCount )
{
    auto res = std::to_string( num );
    if ( res.size() < minDigitCount )
    {
        res.insert( 0, std::string( minDigitCount - res.size(), '0' ) );
    }

    return res;
}

void ImageDecoder::Attach(std::shared_ptr<std::istream> pStream)
{
    if (!pStream)
        throw std::invalid_argument("pStream is null");

    _pStream = pStream;
}

void ImageDecoder::Attach(const std::string &fileName)
{
    _lastFileName = fileName;
    std::shared_ptr<std::ifstream> pStream(new std::ifstream(fileName, std::ios_base::in | std::ios_base::binary));
    if (!pStream->is_open())
        throw std::invalid_argument("fileName");

    Attach(pStream);
}

void ImageDecoder::Reattach()
{
    Detach();
    if (!_lastFileName.empty())
        Attach(_lastFileName);
}

void ImageDecoder::Detach()
{
    _pStream.reset();
}

std::shared_ptr<ImageDecoder> ImageDecoder::Create(const std::string &fileName)
{
    auto path = std::filesystem::path(fileName);
    auto extension = path.extension().string();

    std::transform( extension.begin(), extension.end(), extension.begin(), [](unsigned char c) { return std::tolower(c); });
    std::shared_ptr<ImageDecoder> pDecoder;
    if ( PpmDecoder::GetExtensions().contains( extension ) )
    {
        pDecoder.reset(new PpmDecoder());
    }
    else if ( RawDecoder::GetExtensions().contains( extension ) )
    {
        pDecoder.reset(new RawDecoder());
    }
    else if ( TiffDecoder::GetExtensions().contains( extension ) )
    {
        pDecoder.reset( new TiffDecoder() );
    }

    if (!pDecoder)
        throw std::invalid_argument("fileName");

    pDecoder->Attach(fileName);
    return pDecoder;
}

std::vector<std::shared_ptr<ImageDecoder>> ImageDecoder::GetDecodersFromDir( std::string path )
{
    if ( !std::filesystem::is_directory( path ) )
        return {};

    std::vector<std::shared_ptr<ImageDecoder>>  res;
    for ( const auto& entry : std::filesystem::directory_iterator( path) )
    {         
        if ( std::filesystem::is_directory( entry ) )
            continue;
        
        auto extension = entry.path().extension().string();
        std::transform( extension.begin(), extension.end(), extension.begin(), [] ( unsigned char c ) { return std::tolower( c ); } );
        if ( GetAllExtensions().contains(extension) )
            res.push_back( ImageDecoder::Create( entry.path().string() ) );        
    }

    return res;
}

std::vector<std::shared_ptr<ImageDecoder>> ImageDecoder::GetDecodersFromMask( std::string mask )
{
    size_t start = 0;
    std::vector<std::shared_ptr<ImageDecoder>>  res;

    while ( start < mask.size() )
    {
        auto end = mask.find_first_of( ';', start );
        std::string fileName = mask.substr( start, end - start );
        size_t tildePos = fileName.find_first_of( '~' );

        if ( std::filesystem::exists( fileName ) )
        {
            if ( std::filesystem::is_directory( fileName ) )
            {
                auto decoders = ImageDecoder::GetDecodersFromDir( fileName );
                res.insert( res.end(), decoders.begin(), decoders.end() );
            }
            else
            {
                res.push_back( ImageDecoder::Create( fileName ) );
            }
        }
        else if ( tildePos != std::string::npos )
        {
            size_t pointPos = fileName.find_first_of( '.', tildePos );
            auto varDigitCount = pointPos - tildePos - 1;
            if ( varDigitCount != 0 )
            {

                int minNum = std::stoi( fileName.substr( tildePos - varDigitCount, varDigitCount ) );
                int maxNum = std::stoi( fileName.substr( tildePos + 1, varDigitCount ) );

                for ( int j = minNum; j <= maxNum; ++j )
                {
                    auto tempName = fileName.substr( 0, tildePos - varDigitCount ) + IntToString( j, varDigitCount ) + fileName.substr( pointPos );
                    if ( std::filesystem::exists( tempName ) )
                        res.push_back( Create( tempName ) );
                }
            }
        }

        if ( end == std::string::npos )
            break;

        start = end + 1;
    }

    return res;
}

std::unique_ptr<std::istringstream> ImageDecoder::ReadLine()
{
    std::string res;
    std::getline(*_pStream, res);
    return std::make_unique<std::istringstream>(res);
}

ACMB_NAMESPACE_END