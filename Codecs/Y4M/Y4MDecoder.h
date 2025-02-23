#pragma once
#include "../../Codecs/imagedecoder.h"
#include "../../Core/enums.h"

ACMB_NAMESPACE_BEGIN

namespace Y4M
{
    inline static const size_t cSignatureSize = 8;
}

class Y4MDecoder : public ImageDecoder
{
    size_t _frameBufferSize = 0;
    size_t _frameByteSize = 0;
    size_t _headerSize = 0;
    uint32_t _frameRate = 0; 
public:
    using ImageDecoder::Attach;
    /// attach to stream
    void Attach( std::shared_ptr<std::istream> pStream ) override;
    /// read the next frame
    std::shared_ptr<IBitmap> ReadBitmap() override;
    /// read the i-th frame
    std::shared_ptr<IBitmap> ReadBitmap( uint32_t i ) override;
    /// returns supported file extensions
    static std::unordered_set <std::string> GetExtensions();

    Y4MDecoder( PixelFormat outputFormat = PixelFormat::Unspecified );
};

ACMB_NAMESPACE_END
