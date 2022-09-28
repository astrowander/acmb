#pragma once
#include "../../Codecs/imageencoder.h"

ACMB_NAMESPACE_BEGIN

class JpegEncoder : public ImageEncoder
{
    uint8_t _quality = 90;
    bool _downsample = false;

public:
    JpegEncoder( uint8_t quality = 90, bool downsample = false );

    void WriteBitmap( std::shared_ptr<IBitmap> pBitmap ) override;

    static std::unordered_set<std::string> GetExtensions();

    uint8_t GetQuality() const;
    bool GetDownsample() const;

    void SetQuality( uint8_t val );
    void SetDownsample( bool val );

    ADD_EXTENSIONS
};

ACMB_NAMESPACE_END