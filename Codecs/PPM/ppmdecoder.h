#ifndef PPMDECODER_H
#define PPMDECODER_H

#include "../../Codecs/imagedecoder.h"
#include "../../Core/enums.h"

class PpmDecoder : public ImageDecoder
{
    PpmMode _ppmMode = PpmMode::Binary;
    uint32_t _maxval = 0;

    std::streampos _dataOffset = 0;
    uint32_t _currentScanline = 0;

public:
    void Attach(const std::string& fileName) override;
    void Attach(std::shared_ptr<std::istream> pStream) override;

    std::shared_ptr<IBitmap> ReadBitmap() override;
    std::shared_ptr<IBitmap> ReadStripe(uint32_t stripeHeight = 0) override;

    uint32_t GetCurrentScanline() const override;

    static std::unordered_set <std::string> GetExtensions()
    {
        return { ".ppm", ".pgm" };
    }
private:    

    std::shared_ptr<IBitmap> CreateStripe(uint32_t stripeHeight);

    template<uint32_t bytes>
    std::shared_ptr<IBitmap> ReadBinaryStripe(uint32_t stripeHeight);

    std::shared_ptr<IBitmap> ReadTextStripe(uint32_t stripeHeight);

    std::unique_ptr<std::istringstream> ReadLine() override;

    ADD_EXTENSIONS
};

#endif // PPMDECODER_H
