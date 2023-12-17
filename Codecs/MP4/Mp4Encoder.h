#pragma once
#include "../../Codecs/imageencoder.h"

class CJOCh264encoder;

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
    Main,
    Main10,
    MainStillPicture,
    MSP,
    Main_Intra,
    Main10_Intra,
    Main444_8,
    Main444_Intra,
    Main444_StillPicture,
    Main422_10,
    Main422_10_Intra,
    Main444_10,
    Main444_10_Intra,
    Main12,
    Main12_Intra,
    Main422_12,
    Main422_12_Intra,
    Main444_12,
    Main444_12_Intra,
    Main444_16_Intra,
    Main444_16_StillPicture,
    None
};

//struct Mp4EncoderParams;

class Mp4Encoder : public ImageEncoder
{
    //std::shared_ptr<CJOCh264encoder> _pEncoder;
    uint32_t _frameRate = 25;
    std::vector<uint8_t> _yuv;
public:
    Mp4Encoder( /*H265Preset preset, H265Tune tune = H265Tune::None, H265Profile profile = H265Profile::None*/);

    using ImageEncoder::Attach;
    virtual void Attach( std::shared_ptr<std::ostream> pStream ) override;
    /// attach to file
    //virtual void Attach( const std::string& fileName ) override;

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