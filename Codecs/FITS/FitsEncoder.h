#pragma once
#include "../../Codecs/imageencoder.h"

ACMB_NAMESPACE_BEGIN

class FitsEncoder : public ImageEncoder
{
    std::string _fileName;

public:
    FitsEncoder() = default;
    virtual void Attach( std::shared_ptr<std::ostream> pStream ) override;
    /// attach to file
    virtual void Attach( const std::string& fileName ) override;

    virtual void Detach() override;
    /// write given bitmap
    void WriteBitmap( std::shared_ptr<IBitmap> pBitmap ) override;
    /// returns supported extensions
    static std::set<std::string> GetExtensions();

    ADD_EXTENSIONS
};

ACMB_NAMESPACE_END
