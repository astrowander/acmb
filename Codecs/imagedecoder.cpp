#include "imagedecoder.h"

#include "PPM/ppmdecoder.h"
#include "Raw/RawDecoder.h"
#include "Tiff/TiffDecoder.h"
#include "FITS/FitsDecoder.h"
#include "JPEG/JpegDecoder.h"

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

ImageDecoder::ImageDecoder( PixelFormat outputFormat )
{
    _pixelFormat = outputFormat;
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

ImageDecoder::~ImageDecoder()
{
    Detach();
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

std::shared_ptr<ImageDecoder> ImageDecoder::Create(const std::string &fileName, PixelFormat outputFormat )
{
    auto path = std::filesystem::path(fileName);
    auto extension = path.extension().string();

    std::transform( extension.begin(), extension.end(), extension.begin(), [](unsigned char c) { return std::tolower(c); });
    std::shared_ptr<ImageDecoder> pDecoder;
    if ( PpmDecoder::GetExtensions().contains( extension ) )
    {
        pDecoder.reset(new PpmDecoder( outputFormat ));
    }
    else if ( RawDecoder::GetExtensions().contains( extension ) )
    {
        pDecoder.reset(new RawDecoder( outputFormat ));
    }
    else if ( TiffDecoder::GetExtensions().contains( extension ) )
    {
        pDecoder.reset( new TiffDecoder( outputFormat ) );
    }
    else if ( FitsDecoder::GetExtensions().contains( extension ) )
    {
        pDecoder.reset( new FitsDecoder( outputFormat ) );
    }
    else if ( JpegDecoder::GetExtensions().contains( extension ) )
    {
        pDecoder.reset( new JpegDecoder( outputFormat ) );
    }

    if ( !pDecoder )
        throw std::runtime_error( std::string( "unable to open file" ) + fileName );

    pDecoder->Attach(fileName);
    return pDecoder;
}

std::shared_ptr<ImageDecoder> ImageDecoder::Create( std::shared_ptr<std::istream> pStream, PixelFormat outputFormat )
{
    if ( !pStream )
        throw std::invalid_argument("pStream");

    std::shared_ptr<ImageDecoder> pDecoder;
    const char ch = pStream->peek();
    if ( ch == 'P' )
        pDecoder = std::make_shared<PpmDecoder>( outputFormat );
    else if ( ch == 'I' )
        pDecoder = std::make_shared<TiffDecoder>( outputFormat );
    else
        pDecoder = std::make_shared<RawDecoder>( outputFormat );

    pDecoder->Attach(pStream);
    return pDecoder;
}

const std::string& ImageDecoder::GetLastFileName() const
{
    return _lastFileName;
}

std::vector<Pipeline> ImageDecoder::GetPipelinesFromDir( std::string path, PixelFormat outputFormat )
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
        res.emplace_back( ImageDecoder::Create( fileName, outputFormat ) );

    return res;
}

std::vector<Pipeline> ImageDecoder::GetPipelinesFromMask( std::string mask, PixelFormat outputFormat )
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
                auto pipelines = ImageDecoder::GetPipelinesFromDir( fileName, outputFormat );
                res.insert( res.end(), pipelines.begin(), pipelines.end() );
            }
            else
            {
                res.emplace_back( ImageDecoder::Create( fileName, outputFormat ) );
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
                        res.emplace_back( Create( tempName, outputFormat ) );
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
    if ( _pixelFormat == PixelFormat::Unspecified || _pixelFormat == pSrcBitmap->GetPixelFormat() )
        return pSrcBitmap;

    IBitmapPtr pRes = pSrcBitmap;

    if ( pSrcBitmap->GetPixelFormat() == PixelFormat::Bayer16 )
        pRes = DebayerTransform::Debayer( pSrcBitmap, pSrcBitmap->GetCameraSettings() );

    if ( pRes->GetPixelFormat() == _pixelFormat )
        return pRes;

    return Converter::Convert( pRes, _pixelFormat );
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
