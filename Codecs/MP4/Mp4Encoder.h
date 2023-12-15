#pragma once
#include "../../Codecs/imageencoder.h"

ACMB_NAMESPACE_BEGIN

enum class H265Preset
{
    UltraFast,
    SuperFast,
    VeryFast,
    Faster,
    Fast,
    Medium,
    Slower,
    Slow,
    VerySlow
};

enum class H265Tune
{
    Psnr,
    Ssim,
    Grain,
    ZeroLatency,
    FastDecode,
    Animation,
    None
};

enum class H265Profile
{
    Baseline,
    Main,
    High,
    High10,
    High422,
    High444,
    None
};

struct Mp4EncoderParams;

class Mp4Encoder : public ImageEncoder
{
    std::shared_ptr<Mp4EncoderParams> _params;
    FILE* _f = nullptr;

    std::vector<uint8_t> _i420Channels[3];

public:
    Mp4Encoder( H265Preset preset, H265Tune tune = H265Tune::None, H265Profile profile = H265Profile::None );

    virtual void Attach( std::shared_ptr<std::ostream> pStream ) override;
    /// attach to file
    virtual void Attach( const std::string& fileName ) override;

    virtual void Detach() override;
    /// write given bitmap
    void WriteBitmap( std::shared_ptr<IBitmap> pBitmap ) override;
    /// returns supported extensions
    static std::set<std::string> GetExtensions();

    void SetFrameRate( uint32_t frameRate );
    uint32_t GetFrameRate() const;

    ADD_EXTENSIONS
};

ACMB_NAMESPACE_END