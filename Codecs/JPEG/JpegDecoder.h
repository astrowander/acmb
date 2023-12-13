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

public:
    JpegDecoder( PixelFormat outputFormat = PixelFormat::Unspecified );
    void Attach( std::shared_ptr<std::istream> pStream ) override;
    void Detach() override;
    /// read the whole bitmap
    std::shared_ptr<IBitmap> ReadBitmap() override;

    static std::unordered_set <std::string> GetExtensions();

private:
    size_t ReadBytes( JDEC*, uint8_t*, size_t );
    int ReadData( JDEC* jdec, void* data, JRECT* rect );
};

ACMB_NAMESPACE_END