#pragma once
#include "../Core/bitmap.h"
#include <filesystem>
#include <map>
ACMB_NAMESPACE_BEGIN

class SortingHat
{
public:
    struct Frame
    {
        IBitmapPtr pBitmap;
        std::filesystem::path tempFilePath;
        uint32_t index = 0;
        float laplacianMean = 0.0f;
        float laplacianStdDev = 0.0f;
    };

    SortingHat( const ImageParams& imageParams );
    void AddFrame( IBitmapPtr pBitmap );
    std::vector<Frame> GetBestFrames( uint32_t frameCount ) const;
    std::vector<Frame> GetBestFramesByPercentage( float percentage ) const;
    std::vector<Frame> GetBestFramesByQualityThreshold( float qualityThreshold ) const;

    uint32_t GetFrameCount() const;

    const auto& Frames() const
    {
        return _frames;
    }

private:
    //float GetContentRadius( IBitmapPtr pBitmap );
    ImageParams _imageParams;
    std::map<float, Frame, std::greater<float>> _frames;
};

ACMB_NAMESPACE_END
