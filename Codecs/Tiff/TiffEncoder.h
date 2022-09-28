#pragma once

#include "../../Core/enums.h"
#include "../../Codecs/imageencoder.h"
struct TinyTIFFWriterFile;

ACMB_NAMESPACE_BEGIN

class TiffEncoder : public ImageEncoder
{
    std::string _fileName;
    TinyTIFFWriterFile* _pTiff;

public:
    TiffEncoder() = default;

    virtual void Attach( std::shared_ptr<std::ostream> pStream ) override;

    virtual void Attach( const std::string& fileName ) override;

    virtual void Detach() override;

    void WriteBitmap( std::shared_ptr<IBitmap> pBitmap ) override;

    static std::unordered_set<std::string> GetExtensions();

    ADD_EXTENSIONS
};

ACMB_NAMESPACE_END