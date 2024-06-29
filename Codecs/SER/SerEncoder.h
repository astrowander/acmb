#pragma once
#include "../../Codecs/VideoEncoder.h"

ACMB_NAMESPACE_BEGIN

class SerEncoder : public VideoEncoder
{
public:
    using ImageEncoder::Attach;
    virtual void Attach( std::shared_ptr<std::ostream> pStream ) override;

    virtual void Detach() override;
    /// write given bitmap
    void WriteBitmap( std::shared_ptr<IBitmap> pBitmap ) override;
    /// returns supported extensions
    static std::set<std::string> GetExtensions();

    ADD_EXTENSIONS
};

ACMB_NAMESPACE_END
