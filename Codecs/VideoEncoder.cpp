#include "VideoEncoder.h"
#include "../Core/bitmap.h"

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

ACMB_NAMESPACE_BEGIN

void VideoEncoder::SetFrameRate( uint32_t rate )
{
    _frameRate = rate;
}

uint32_t VideoEncoder::GetFrameRate() const
{
    return _frameRate;
}

void VideoEncoder::SetTotalFrames( uint32_t frames )
{
    _totalFrames = frames;
}

uint32_t VideoEncoder::GetTotalFrames() const
{
    return _totalFrames;
}

ACMB_NAMESPACE_END