#ifndef REGISTRATOR_H
#define REGISTRATOR_H
#include "Tests/testtools.h"
#include "Transforms/converter.h"
#include "Codecs/imagedecoder.h"
#include <algorithm>

#include "star.h"

#include "alignmentdataset.h"

//class IBitmap;
class Registrator
{
    std::shared_ptr<AlignmentDataset> _dataset;
    std::shared_ptr<IBitmap> _pBitmap;
    double _threshold;
    uint32_t _minStarSize;
    uint32_t _maxStarSize;
    //std::vector<Star> _stars;

private:
    Registrator(std::shared_ptr<IBitmap> pBitmap, double threshold, uint32_t minStarSize, uint32_t maxStarSize);

    template <PixelFormat pixelFormat>
    void Registrate()
    {
        using ChannelType = typename PixelFormatTraits<pixelFormat>::ChannelType;
        auto pGrayBitmap = std::static_pointer_cast<Bitmap<pixelFormat>>(Convert(_pBitmap, pixelFormat));
        //IBitmap::Save(pGrayBitmap, GetPathToTestFile("gray.pgm"));
        auto w = pGrayBitmap->GetWidth();
        auto h = pGrayBitmap->GetHeight();

        auto data = std::vector<ChannelType>(pGrayBitmap->GetScanline(0), pGrayBitmap->GetScanline(0) + w * h);
        auto median = data.begin() + data.size() / 2;
        std::nth_element(data.begin(), median, data.end());

        auto threshold = static_cast<ChannelType>(std::min(static_cast<uint32_t>(*median * (1 + _threshold / 100)), static_cast<uint32_t>(std::numeric_limits<ChannelType>::max())));

        auto pData = pGrayBitmap->GetScanline(0);
        for (uint32_t i = 0; i < h; ++i)
        {
            for (uint32_t j = 0; j < w; ++j)
            {
                if (pData[i * w + j] > threshold)
                {
                    Star star {Rect {static_cast<int32_t>(j), static_cast<int32_t>(i), 1, 1}, 0, 0, 0};
                    InspectStar(star, threshold, pData, j, i, w, h);
                    if (star.rect.width >= _minStarSize && star.rect.width <= _maxStarSize && star.rect.height >= _minStarSize && star.rect.height <= _maxStarSize)
                    {
                        star.center.x /= star.luminance;
                        star.center.y /= star.luminance;
                        _dataset->stars.push_back(star);
                    }
                }
            }

        }

        //IBitmap::Save(pGrayBitmap, GetPathToTestFile("mask.pgm"));
    }

    template <typename ChannelType>
    void InspectStar(Star& star, ChannelType threshold, ChannelType* pData, uint32_t x, uint32_t y, uint32_t w, uint32_t h)
    {
        ++star.pixelCount;        
        auto pixelLuminance = pData[y * w + x] - threshold;
        star.luminance += pixelLuminance;
        star.center.x += x * pixelLuminance;
        star.center.y += y * pixelLuminance;
        pData[y * w + x] = 0;

        if (x + 1 < w && pData[y * w + x + 1] > threshold)
        {
            star.rect.ExpandRight(x + 1);
            InspectStar(star, threshold, pData, x + 1, y, w, h);
        }

        if (x + 1 < w && y + 1 < h && pData[(y + 1) * w + x + 1] > threshold)
        {
            star.rect.ExpandRight(x + 1);
            star.rect.ExpandDown(y + 1);
            InspectStar(star, threshold, pData, x + 1, y + 1, w, h);
        }

        if (y + 1 < h && pData[(y + 1) * w + x] > threshold)
        {
            star.rect.ExpandDown(y + 1);
            InspectStar(star, threshold, pData, x, y + 1, w, h);
        }

        if (x > 0 && y + 1 < h && pData[(y + 1) * w + x - 1] > threshold)
        {
            star.rect.ExpandDown(y + 1);
            star.rect.ExpandLeft(x - 1);
            InspectStar(star, threshold, pData, x - 1, y + 1, w, h);
        }
    }

public:
    static std::shared_ptr<AlignmentDataset> Registrate(std::shared_ptr<IBitmap> pBitmap, double threshold = 10, uint32_t minStarSize = 3, uint32_t maxStarSize = 20);
};

#endif
