#pragma once
#include "../Transforms/converter.h"
#include "../Codecs/imagedecoder.h"

#include <algorithm>
#include <cstring>
#include "star.h"

ACMB_NAMESPACE_BEGIN
/// <summary>
/// Find stars on image and returns them in a list
/// </summary>
class Registrator
{
public:
    /// split image to squares of this size and detects stars in parallel
    static constexpr uint32_t tileSize = 600;

private:
    std::shared_ptr<IBitmap> _pBitmap;
    double _threshold = 40;
    int _minStarSize = 5;
    int _maxStarSize = 25;

    std::vector<std::vector<Star>> _stars;

public:
    Registrator() = default;
    /// Creates registarator with given threshold (star will be detected if its luminosity more than threshold percent above median level)
    /// Also star will be placed to list only if its size more than minStarSize and less than maxStarSize
    Registrator( double threshold, uint32_t _minStarSize = 5, uint32_t _maxStarSize = 25);
    /// Detects stars in the given image
    void Registrate(std::shared_ptr<IBitmap> pBitmap);
    /// Returns list of detected stars
    const std::vector<std::vector<Star>>& GetStars() const;   
    
private:

    template <PixelFormat pixelFormat>
    std::vector<Star> Registrate(Rect roi)
    {
        std::vector<Star> res;

        using ChannelType = typename PixelFormatTraits<pixelFormat>::ChannelType;
        auto pGrayBitmap = std::static_pointer_cast<Bitmap<pixelFormat>>(_pBitmap);

        auto w = pGrayBitmap->GetWidth();
        auto h = pGrayBitmap->GetHeight();

        auto data = std::vector<ChannelType>(roi.width * roi.height * ChannelCount(pixelFormat));
        for (int i = 0; i < roi.height; ++i)
        {
            memcpy(&data[i * roi.width], pGrayBitmap->GetScanline(i + roi.y) + roi.x, roi.width * BytesPerPixel(pixelFormat));
        }

        auto median = data.begin() + data.size() / 2;
        std::nth_element(data.begin(), median, data.end());

        auto threshold = static_cast<ChannelType>(std::min(static_cast<uint32_t>(*median * (1 + _threshold / 100)), static_cast<uint32_t>(std::numeric_limits<ChannelType>::max())));

        auto pData = pGrayBitmap->GetScanline(0);
        

        for (int i = roi.y; i < roi.y + roi.height; ++i)
        {
            for (int j = roi.x; j < roi.x + roi.width; ++j)
            {
                if (pData[i * w + j] > threshold)
                {
                    Star star {Rect {static_cast<int32_t>(j), static_cast<int32_t>(i), 1, 1}, {0, 0}, 0};
                    InspectStar(star, threshold, pData, j, i, w, h, roi);
                    if (star.rect.width >= _minStarSize && star.rect.width <= _maxStarSize && star.rect.height >= _minStarSize && star.rect.height <= _maxStarSize)
                    {
                        star.center.x /= star.luminance;
                        star.center.y /= star.luminance;
                        res.push_back(star);
                    }
                }
            }

        }

        return res;
    }

    template <typename ChannelType>
    void InspectStar(Star& star, ChannelType threshold, ChannelType* pData, int x, int y, int w, int h, Rect roi)
    {
        ++star.pixelCount;        
        auto pixelLuminance = pData[y * w + x] - threshold;
        star.luminance += pixelLuminance;
        star.center.x += x * pixelLuminance;
        star.center.y += y * pixelLuminance;
        pData[y * w + x] = 0;

        if (star.rect.width > _maxStarSize + 1 || star.rect.height > _maxStarSize + 1)
            return;

        if (x + 1 < roi.x + roi.width && pData[y * w + x + 1] > threshold)
        {
            star.rect.ExpandRight(x + 1);
            InspectStar(star, threshold, pData, x + 1, y, w, h, roi);
        }

        if (x + 1 < roi.x + roi.width && y + 1 < roi.y + roi.height && pData[(y + 1) * w + x + 1] > threshold)
        {
            star.rect.ExpandRight(x + 1);
            star.rect.ExpandDown(y + 1);
            InspectStar(star, threshold, pData, x + 1, y + 1, w, h, roi);
        }

        if (y + 1 < roi.y + roi.height && pData[(y + 1) * w + x] > threshold)
        {
            star.rect.ExpandDown(y + 1);
            InspectStar(star, threshold, pData, x, y + 1, w, h, roi);
        }

        if (x > roi.x && y + 1 < roi.y + roi.height && pData[(y + 1) * w + x - 1] > threshold)
        {
            star.rect.ExpandDown(y + 1);
            star.rect.ExpandLeft(x - 1);
            InspectStar(star, threshold, pData, x - 1, y + 1, w, h, roi);
        }
    }
};

ACMB_NAMESPACE_END
