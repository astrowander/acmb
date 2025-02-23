#pragma once
#include "imageencoder.h"

ACMB_NAMESPACE_BEGIN

class IBitmap;

class VideoEncoder : public ImageEncoder
{
protected:
    uint32_t _frameRate = 25;
    uint32_t _totalFrames = 0;
    std::vector<uint8_t> _yuv;


public:
    void SetFrameRate( uint32_t frameRate );
    uint32_t GetFrameRate() const;

    void SetTotalFrames( uint32_t totalFrames );
    uint32_t GetTotalFrames() const;
};
ACMB_NAMESPACE_END