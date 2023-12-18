#pragma once
#include "imageencoder.h"

ACMB_NAMESPACE_BEGIN

class IBitmap;

class VideoEncoder : public ImageEncoder
{
protected:
    uint32_t _frameRate = 25;
    std::vector<uint8_t> _yuv;

    void BitmapToYuv( std::shared_ptr<IBitmap> pBitmap );

public:
    void SetFrameRate( uint32_t frameRate );
    uint32_t GetFrameRate() const;
};
ACMB_NAMESPACE_END