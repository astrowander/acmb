#pragma once

#include "../imagedecoder.h"
#include "../../Core/enums.h"

class LibRaw;

ACMB_NAMESPACE_BEGIN
/// Settings for RAW files reading
struct RawSettings
{
    /// if true bitmap size will be two times smaller
    bool halfSize = false;
    /// if format is grayscale picture will not be debayered
    PixelFormat outputFormat = PixelFormat::RGB48;
};

/// <summary>
/// Reads RAW files
/// </summary>
class RawDecoder : public ImageDecoder
{
	LibRaw* _pLibRaw;
    RawSettings _rawSettings;

public:
    /// Creates riader with given settings
    RawDecoder( const RawSettings& rawSettings = {} );
    ~RawDecoder();
    /// attach to a file
    void Attach(const std::string& fileName) override;
    /// attach to a stream
    void Attach(std::shared_ptr<std::istream> pStream) override;
    void Detach() override;
    ///read whole bitmap
    std::shared_ptr<IBitmap> ReadBitmap() override;
    ///returns supported extensions
    static std::unordered_set <std::string> GetExtensions();

    ADD_EXTENSIONS
};

ACMB_NAMESPACE_END