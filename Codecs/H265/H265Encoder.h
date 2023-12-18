#pragma once
#include "./../VideoEncoder.h"

ACMB_NAMESPACE_BEGIN

struct H265EncoderParams;

class H265Encoder : public VideoEncoder
{
    std::shared_ptr<H265EncoderParams> _params;

public:
    enum class Preset
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

    enum class Tune
    {
        Psnr,
        Ssim,
        Grain,
        ZeroLatency,
        FastDecode,
        Animation,
        None
    };

    enum class Profile
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

    H265Encoder( Preset preset, Tune tune = Tune::None, Profile profile = Profile::None );

    using ImageEncoder::Attach;
    virtual void Detach() override;

    virtual void WriteBitmap( std::shared_ptr<IBitmap> pBitmap ) override;
    static std::set<std::string> GetExtensions();

    ADD_EXTENSIONS
};

ACMB_NAMESPACE_END