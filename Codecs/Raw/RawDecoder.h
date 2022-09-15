#ifndef RAWDECODER_H
#define RAWDECODER_H

#include "../imagedecoder.h"
#include "../../Core/enums.h"

class LibRaw;

class RawDecoder : public ImageDecoder
{
	LibRaw* _pLibRaw;
    bool _halfSize;

public:
	RawDecoder(bool halfSize = false);
    ~RawDecoder();
    void Attach(const std::string& fileName) override;
    void Attach(std::shared_ptr<std::istream> pStream) override;
    void Detach() override;

    std::shared_ptr<IBitmap> ReadBitmap() override;
    std::shared_ptr<IBitmap> ReadStripe(uint32_t stripeHeight = 0) override;

    uint32_t GetCurrentScanline() const override;

    static std::unordered_set <std::string> GetExtensions()
    {
        return { ".cr2", ".dng" };
    }

    ADD_EXTENSIONS
};

#endif