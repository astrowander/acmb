#pragma once

#include "../imagedecoder.h"
#include "../../Core/enums.h"

class LibRaw;

ACMB_NAMESPACE_BEGIN

struct RawSettings
{
    bool halfSize = false;
    bool extendedFormat = true;
};

class RawDecoder : public ImageDecoder
{
	LibRaw* _pLibRaw;
    RawSettings _rawSettings;

public:
    RawDecoder( const RawSettings& rawSettings = {} );
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

ACMB_NAMESPACE_END