#include "imagedecoder.h"

#include "PPM/ppmdecoder.h"
#include "Raw/RawDecoder.h"
#include "Tiff/TiffDecoder.h"

#include "./../Transforms/DebayerTransform.h"
#include "./../Transforms/converter.h"

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

ImageDecoder::ImageDecoder( const DecoderSettings& settings )
: _decoderSettings( settings )
{
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

std::shared_ptr<IBitmap> ImageDecoder::ReadStripe( uint32_t )
{
    throw std::runtime_error( "not implemented" );
}

uint32_t ImageDecoder::GetCurrentScanline() const
{
    throw std::runtime_error( "not implemented" );
}

IBitmapPtr ImageDecoder::ProcessBitmap( IBitmapPtr )
{
    return ReadBitmap();
}

std::shared_ptr<ImageDecoder> ImageDecoder::Create(const std::string &fileName, const DecoderSettings& rawSettings )
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
        pDecoder.reset(new RawDecoder( rawSettings ));
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

const std::string& ImageDecoder::GetLastFileName() const
{
    return _lastFileName;
}

std::vector<Pipeline> ImageDecoder::GetPipelinesFromDir( std::string path, const DecoderSettings& rawSettings )
{
    if ( !std::filesystem::is_directory( path ) )
        return {};

    std::vector<Pipeline>  res;
    std::set<std::string> sortedNames;
    for ( const auto& entry : std::filesystem::directory_iterator( path) )
    {         
        if ( std::filesystem::is_directory( entry ) )
            continue;
        
        auto extension = entry.path().extension().string();
        std::transform( extension.begin(), extension.end(), extension.begin(), [] ( unsigned char c ) { return std::tolower( c ); } );
        if ( GetAllExtensions().contains(extension) )
            sortedNames.insert( entry.path().string() );
    }

    for (const auto & fileName : sortedNames)
        res.emplace_back( ImageDecoder::Create( fileName, rawSettings ) );

    return res;
}

std::vector<Pipeline> ImageDecoder::GetPipelinesFromMask( std::string mask, const DecoderSettings& rawSettings )
{
    size_t start = 0;
    std::vector<Pipeline> res;

    while ( start < mask.size() )
    {
        auto end = mask.find_first_of( ';', start );
        std::string fileName = mask.substr( start, end - start );
        size_t separatorPos = fileName.find_first_of( '#' );

        if ( std::filesystem::exists( fileName ) )
        {
            if ( std::filesystem::is_directory( fileName ) )
            {
                auto pipelines = ImageDecoder::GetPipelinesFromDir( fileName, rawSettings );
                res.insert( res.end(), pipelines.begin(), pipelines.end() );
            }
            else
            {
                res.emplace_back( ImageDecoder::Create( fileName, rawSettings ) );
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
                        res.emplace_back( Create( tempName, rawSettings ) );
                }
            }
        }

        if ( end == std::string::npos )
            break;

        start = end + 1;
    }

    return res;
}

const std::unordered_set<std::string>& ImageDecoder::GetAllExtensions()
{
    return _allExtensions;
}

IBitmapPtr ImageDecoder::ToOutputFormat( IBitmapPtr pSrcBitmap )
{
    if ( _decoderSettings.outputFormat == PixelFormat::Unspecified || _decoderSettings.outputFormat == pSrcBitmap->GetPixelFormat() )
        return pSrcBitmap;

    IBitmapPtr pRes;

    if ( pSrcBitmap->GetPixelFormat() == PixelFormat::Bayer16 )
        pRes = DebayerTransform::Debayer( pSrcBitmap, pSrcBitmap->GetCameraSettings() );

    if ( pRes->GetPixelFormat() == _decoderSettings.outputFormat )
        return pRes;

    return Converter::Convert( pRes, _decoderSettings.outputFormat );
}

bool ImageDecoder::AddCommonExtensions( const std::unordered_set<std::string>& extensions )
{
    _allExtensions.insert( std::begin( extensions ), std::end( extensions ) );
    return true;
}

std::unique_ptr<std::istringstream> ImageDecoder::ReadLine()
{
    std::string res;
    std::getline(*_pStream, res);
    return std::make_unique<std::istringstream>(res);
}

ACMB_NAMESPACE_END
