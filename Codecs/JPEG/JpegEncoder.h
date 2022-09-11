#pragma once
#include "../../Codecs/imageencoder.h"

class JpegEncoder : public ImageEncoder
{
    uint8_t _quality = 90;
    bool _downsample = false;

public:
    JpegEncoder( uint8_t quality = 90, bool downsample = false );

    void WriteBitmap( std::shared_ptr<IBitmap> pBitmap ) override;

    static std::unordered_set<std::string> GetExtensions()
    {
        return { ".jpg", ".jpeg", ".jfif" };
    }

    uint8_t GetQuality();
    bool GetDownsample();

    void SetQuality( uint8_t val );
    void SetDownsample( bool val );

    ADD_EXTENSIONS
};