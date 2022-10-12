#pragma once

#include "../imagedecoder.h"
#include "../../Core/enums.h"
#include <unordered_map>

class LibRaw;

ACMB_NAMESPACE_BEGIN

struct LensInfo
{
    std::string fullName;    
    int minFocal = 0;
    int maxFocal = 0;
    LensInfo() = default;
};

/// <summary>
/// Reads RAW files
/// </summary>
class RawDecoder : public ImageDecoder
{
    using LensDB = std::unordered_map<uint16_t, std::vector<LensInfo>>;
    static LensDB LoadLensDB();
    inline static LensDB lensDB = LoadLensDB();
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

    const RawSettings& GetRawSettings();
    void SetRawSettings( const RawSettings& rawSettings );

    ADD_EXTENSIONS
};

ACMB_NAMESPACE_END