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

    SortingHat( const ImageParams& imageParams, float percentageToSelect = 25.0f );
    void AddFrame( IBitmapPtr pBitmap );
    std::vector<Frame> GetBestFrames() const;

    const auto& Frames() const
    {
        return _frames;
    }

private:
    //float GetContentRadius( IBitmapPtr pBitmap );
    ImageParams _imageParams;
    std::map<float, Frame, std::greater<float>> _frames;
    float _percentageToSelect = 25.0f;
};

ACMB_NAMESPACE_END
