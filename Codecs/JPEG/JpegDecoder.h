#pragma once
#include "../imagedecoder.h"
#include "../../Core/bitmap.h"

struct JDEC;
struct JRECT;

ACMB_NAMESPACE_BEGIN

class JpegDecoder : public ImageDecoder
{
    std::vector<uint8_t> _jdWorkspace;
    std::shared_ptr<JDEC> _pJdec;
    std::shared_ptr<Bitmap<PixelFormat::RGB24>> _pBitmap;
    size_t _startDataPos{};

public:
    
    using ImageDecoder::Attach;
    JpegDecoder( PixelFormat outputFormat = PixelFormat::Unspecified );
    void Attach( std::shared_ptr<std::istream> pStream ) override;
    void Detach() override;
    /// read the whole bitmap
    std::shared_ptr<IBitmap> ReadBitmap() override;

    static std::unordered_set <std::string> GetExtensions();
};

ACMB_NAMESPACE_END