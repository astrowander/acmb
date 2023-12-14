#pragma once
#include "../../Codecs/imageencoder.h"

ACMB_NAMESPACE_BEGIN

enum class H264Preset
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

enum class H264Tune
{
    Film,
    Animation,
    Grain,
    StillImage,
    Psnr,
    Ssim,
    FastDecode,
    ZeroLatency,
    None
};

enum class H264Profile
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

public:
    Mp4Encoder( H264Preset preset, H264Tune tune = H264Tune::None, H264Profile profile = H264Profile::None );

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