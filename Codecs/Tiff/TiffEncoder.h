#pragma once

#include "../../Core/enums.h"
#include "../../Codecs/imageencoder.h"
struct TinyTIFFWriterFile;

ACMB_NAMESPACE_BEGIN
/// <summary>
/// Writes bitmaps in TIFF format without compression and tiling
/// </summary>
class TiffEncoder : public ImageEncoder
{
    std::string _fileName;
    TinyTIFFWriterFile* _pTiff;

public:
    TiffEncoder() = default;
    /// attach to stream
    virtual void Attach( std::shared_ptr<std::ostream> pStream ) override;
    /// attach to file
    virtual void Attach( const std::string& fileName ) override;

    virtual void Detach() override;
    /// write given bitmap
    void WriteBitmap( std::shared_ptr<IBitmap> pBitmap ) override;
    /// returns supported extensions
    static std::unordered_set<std::string> GetExtensions();

    ADD_EXTENSIONS
};

ACMB_NAMESPACE_END