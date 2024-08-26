#pragma once
#include "../../Codecs/imagedecoder.h"
#include "../../Core/enums.h"

namespace Ser
{
    enum class ColorID
    {
        MONO = 0,
        BAYER_RGGB = 8,
        BAYER_GRBG = 9,
        BAYER_GBRG = 10,
        BAYER_BGGR = 11,
        BAYER_CYYM = 16,
        BAYER_YCMY = 17,
        BAYER_YMCY = 18,
        BAYER_MYYC = 19,
        RGB = 100,
        BGR = 101
    };

    inline static const size_t cHeaderSize = 178;
}

ACMB_NAMESPACE_BEGIN

class SerDecoder : public ImageDecoder
{
    /*struct Header
    {
        char fileID[14];
        int32_t luID;
        ColorID colorID;
        int32_t littleEndian;
        int32_t imageWidth;
        int32_t imageHeight;
        int32_t pixelDepthPerPlane;
        int32_t frameCount;
        char observer[40];
        char instrument[40];
        char telescope[40];
        int64_t dateTime;
        int64_t dateTimeUTC;
    };*/

    //Header header;
    size_t _frameByteSize = 0;

public:
    SerDecoder( PixelFormat outputFormat = PixelFormat::Unspecified );

    using ImageDecoder::Attach;
    /// attach to stream
    void Attach( std::shared_ptr<std::istream> pStream ) override;
    /// read the next frame
    std::shared_ptr<IBitmap> ReadBitmap() override;
    /// read the i-th frame
    std::shared_ptr<IBitmap> ReadBitmap( uint32_t i ) override;
    /// returns supported file extensions
    static std::unordered_set <std::string> GetExtensions();
};

ACMB_NAMESPACE_END
