#pragma once

#include "../../Codecs/imagedecoder.h"
#include "../../Core/enums.h"

ACMB_NAMESPACE_BEGIN
/// <summary>
/// Reads PPM/PGM files
/// </summary>
class PpmDecoder : public ImageDecoder
{
    PpmMode _ppmMode = PpmMode::Binary;
    uint32_t _maxval = 0;

    std::streampos _dataOffset = 0;
    uint32_t _currentScanline = 0;

public:
    PpmDecoder( PixelFormat outputFormat = PixelFormat::Unspecified );
    /// attach to file
    void Attach(const std::string& fileName) override;
    /// attach to stream
    void Attach(std::shared_ptr<std::istream> pStream) override;
    /// read the whole bitmap
    std::shared_ptr<IBitmap> ReadBitmap() override;
    /// read several lines. If stripeHeight==0 read the whole bitmap
    std::shared_ptr<IBitmap> ReadStripe(uint32_t stripeHeight = 0) override;
    /// returns first scanline of the next stripe
    uint32_t GetCurrentScanline() const override;
    /// returns supported file extensions
    static std::unordered_set <std::string> GetExtensions();
private:    

    std::shared_ptr<IBitmap> CreateStripe(uint32_t stripeHeight);

    template<uint32_t bytes>
    std::shared_ptr<IBitmap> ReadBinaryStripe(uint32_t stripeHeight);

    std::shared_ptr<IBitmap> ReadTextStripe(uint32_t stripeHeight);

    std::unique_ptr<std::istringstream> ReadLine() override;

    ADD_EXTENSIONS
};

ACMB_NAMESPACE_END
