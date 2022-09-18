#pragma once

#include "../imagedecoder.h"
#include "../../Core/bitmap.h"

struct TinyTIFFReaderFile;

ACMB_NAMESPACE_BEGIN

class TiffDecoder : public ImageDecoder
{
    TinyTIFFReaderFile* _pReader;

   

public:
    TiffDecoder() = default;

    void Attach( const std::string& fileName ) override;
    void Attach( std::shared_ptr<std::istream> pStream ) override;
    void Detach() override;

    std::shared_ptr<IBitmap> ReadBitmap() override;
    std::shared_ptr<IBitmap> ReadStripe( uint32_t stripeHeight = 0 ) override;

    uint32_t GetCurrentScanline() const override;

    static std::unordered_set <std::string> GetExtensions()
    {
        return { ".tiff", ".tif" };
    }

    ADD_EXTENSIONS
};

ACMB_NAMESPACE_END