#pragma once
#include "../../Codecs/imageencoder.h"
#include "x264.h"

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

class Mp4Encoder : public ImageEncoder
{
    x264_param_t _param;
    x264_picture_t _pic;
    x264_picture_t _pic_out;
    x264_t* _h = nullptr;
    x264_nal_t* _nal = nullptr;
    int i_nal = 0;
    int _i_frame = 0;

    FILE* _f;

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

    ADD_EXTENSIONS
};

ACMB_NAMESPACE_END