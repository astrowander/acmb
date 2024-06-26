#include "imageencoder.h"
#include "./../Core/bitmap.h"
#include <fstream>
#include <filesystem>
#include "PPM/ppmencoder.h"
#include "Tiff/TiffEncoder.h"
#include "JPEG/JpegEncoder.h"
#include "FITS/FitsEncoder.h"
#include "Y4M/Y4MEncoder.h"
#include "H265/H265Encoder.h"
#include "SER/SerEncoder.h"

ACMB_NAMESPACE_BEGIN

void ImageEncoder::Attach(std::shared_ptr<std::ostream> pStream)
{
    if (!pStream)
        throw std::invalid_argument("pStream is null");

    _pStream = pStream;
}

void ImageEncoder::Attach(const std::string &fileName)
{
    std::shared_ptr<std::ofstream> pStream(new std::ofstream(fileName, std::ios_base::binary | std::ios_base::out));
    if (!pStream->is_open())
        throw std::runtime_error("file open error");

    Attach(pStream);
}

void ImageEncoder::Detach()
{
    _pStream.reset();
}

ImageEncoder::~ImageEncoder()
{
    Detach();
}

std::shared_ptr<ImageEncoder> ImageEncoder::Create(const std::string &fileName)
{
    auto path = std::filesystem::path(fileName);
    auto extension = path.extension().string();
    std::shared_ptr<ImageEncoder> pEncoder;
    if (PpmEncoder::GetExtensions().contains(extension))
    {
        pEncoder.reset(new PpmEncoder(PpmMode::Binary));
    }
    else if ( TiffEncoder::GetExtensions().contains( extension ) )
    {
        pEncoder.reset( new TiffEncoder() );
    }
    else if ( JpegEncoder::GetExtensions().contains( extension ) )
    {
        pEncoder.reset( new JpegEncoder() );
    }
    else if ( FitsEncoder::GetExtensions().contains( extension ) )
    {
        pEncoder.reset( new FitsEncoder() );
    }
    else if ( Y4MEncoder::GetExtensions().contains( extension ) )
    {
        pEncoder.reset( new Y4MEncoder() );
    }
    else if ( H265Encoder::GetExtensions().contains( extension ) )
    {
        pEncoder.reset( new H265Encoder( H265Encoder::Preset::VeryFast, H265Encoder::Tune::FastDecode, H265Encoder::Profile::Main ) );
    }
    else if ( SerEncoder::GetExtensions().contains( extension ) )
    {
        pEncoder.reset( new SerEncoder() );
    }

    if (!pEncoder)
        throw std::invalid_argument("unsupported file extension");

    pEncoder->Attach( fileName );
    return pEncoder;
}

const std::set<std::string>& ImageEncoder::GetAllExtensions()
{
    return _allExtensions;
}

IBitmapPtr ImageEncoder::ProcessBitmap( IBitmapPtr pBitmap )
{
    WriteBitmap( pBitmap );
    return pBitmap;
}

bool ImageEncoder::AddCommonExtensions( const std::set<std::string>& extensions )
{
    _allExtensions.insert( std::begin( extensions ), std::end( extensions ) );
    return true;
}

ACMB_NAMESPACE_END
