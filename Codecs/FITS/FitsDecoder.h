#pragma once

#include "../imagedecoder.h"
#include "../../Core/bitmap.h"

namespace CCfits
{
    class FITS;
}

ACMB_NAMESPACE_BEGIN

class FitsDecoder : public ImageDecoder
{
    std::unique_ptr<CCfits::FITS> _pFits;

public:
    FitsDecoder( PixelFormat outputFormat = PixelFormat::Unspecified );
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
