#include "imagedecoder.h"
#include <fstream>
#include <filesystem>
#include <set>

#include "PPM/ppmdecoder.h"
#include "RAW/rawdecoder.h"
#include <filesystem>

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

const std::set<std::string> rawExtensions = { ".cr2", ".CR2", ".dng" };

std::shared_ptr<ImageDecoder> ImageDecoder::Create(const std::string &fileName)
{
    auto path = std::filesystem::path(fileName);
    auto extension = path.extension();
    std::shared_ptr<ImageDecoder> pDecoder;
    if (extension == ".pgm" || extension == ".ppm")
    {
        pDecoder.reset(new PpmDecoder());
    }
    else if (rawExtensions.find(extension.generic_string()) != std::end(rawExtensions))
    {
        pDecoder.reset(new RawDecoder());
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
        if ( !std::filesystem::is_directory( entry ) )
            res.push_back( ImageDecoder::Create( entry.path().u8string() ) );
    }

    return res;
}

std::unique_ptr<std::istringstream> ImageDecoder::ReadLine()
{
    std::string res;
    std::getline(*_pStream, res);
    return std::make_unique<std::istringstream>(res);
}
