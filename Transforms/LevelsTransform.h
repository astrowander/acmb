#pragma once
#include "basetransform.h"
ACMB_NAMESPACE_BEGIN

class ChannelEqualizer;

class LevelsTransform : public BaseTransform
{
protected:
    LevelsTransform( IBitmapPtr pSrcBitmap );

    std::shared_ptr<ChannelEqualizer> _pCommonEqualizer;
    std::shared_ptr<ChannelEqualizer> _pPerChannelEqualizer;

public:
    struct ChannelLevels
    {
        float min = 0.0f;
        float gamma = 1.0f;
        float max = 1.0f;
    };

    struct Settings
    {
        std::array<ChannelLevels, 4> levels;
        bool adjustChannels = false;
    };

    /// Creates instance with source bitmap and given channel transformations. Number of transforms must be equal to the number of channels
    static std::shared_ptr<LevelsTransform> Create( IBitmapPtr pSrcBitmap, const Settings& levels );

    /// Creates instance with source bitmap and given channel transformations. Number of transforms must be equal to the number of channels
    static std::shared_ptr<LevelsTransform> Create( PixelFormat pixelFormat, const Settings& levels );

    static IBitmapPtr ApplyLevels( IBitmapPtr pSrcBitmap, const Settings& levels );

    static Settings GetAutoSettings( IBitmapPtr pSrcBitmap, bool adjustChannels = false );
};

ACMB_NAMESPACE_END

