#pragma once

#include "../imagedecoder.h"
#include "../../Core/bitmap.h"

struct TinyTIFFReaderFile;

ACMB_NAMESPACE_BEGIN
/// <summary>
/// Reads simple TIFF files without compression and tiling
/// </summary>
class TiffDecoder : public ImageDecoder
{
    TinyTIFFReaderFile* _pReader;

public:
    TiffDecoder( const DecoderSettings& settings = {} );
    /// attach to file
    void Attach( const std::string& fileName ) override;
    /// attach to stream
    void Attach( std::shared_ptr<std::istream> pStream ) override;
    void Detach() override;
    /// read whole bitmap
    std::shared_ptr<IBitmap> ReadBitmap() override;
    /// returns supported extensions
    static std::unordered_set <std::string> GetExtensions();

    ADD_EXTENSIONS
};

ACMB_NAMESPACE_END