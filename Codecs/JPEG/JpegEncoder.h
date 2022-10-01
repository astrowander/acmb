#pragma once
#include "../../Codecs/imageencoder.h"

ACMB_NAMESPACE_BEGIN
/// <summary>
/// Writes JPEG file with specified quality and optional downsampling
/// </summary>
class JpegEncoder : public ImageEncoder
{
    uint8_t _quality = 90;
    bool _downsample = false;

public:
    /// <param name="quality">sets compession level, from 0 to 100</param>
    /// <param name="downsample">enable or disable downsampling</param>
    JpegEncoder( uint8_t quality = 90, bool downsample = false );
    /// Writes given bitmap
    void WriteBitmap( std::shared_ptr<IBitmap> pBitmap ) override;
    /// Returns supported file extensions
    static std::unordered_set<std::string> GetExtensions();

    uint8_t GetQuality() const;
    bool GetDownsample() const;

    void SetQuality( uint8_t val );
    void SetDownsample( bool val );

    ADD_EXTENSIONS
};

ACMB_NAMESPACE_END