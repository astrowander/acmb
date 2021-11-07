#ifndef REGISTRATOR_H
#define REGISTRATOR_H
#include "../Tests/testtools.h"
#include "../Transforms/converter.h"
#include "../Codecs/imagedecoder.h"
#include <algorithm>

#include "star.h"

#include "alignmentdataset.h"

//class IBitmap;
struct DatasetTiles
{
    std::vector<std::shared_ptr<AlignmentDataset>> datasets;
    uint32_t totalStarCount = 0;
};

class Registrator
{
    std::shared_ptr<IBitmap> _pBitmap;
    std::vector<bool> _visitedPixels;
    double _threshold;
    uint32_t _minStarSize;
    uint32_t _maxStarSize;
    uint32_t _hTiles;
    uint32_t _vTiles;

public:
    Registrator(double threshold = 40, uint32_t minStarSize = 5, uint32_t maxStarSize = 25, uint32_t hTiles = 1, uint32_t vTiles = 1);

    std::shared_ptr<DatasetTiles> Registrate(std::shared_ptr<IBitmap> pBitmap);

private:
    template <PixelFormat pixelFormat>
    std::shared_ptr<AlignmentDataset> Registrate(Rect roi)
    {
        auto res = std::make_shared<AlignmentDataset>();
        _visitedPixels = std::vector<bool>(_pBitmap->GetWidth() * _pBitmap->GetHeight(), false);

        using ChannelType = typename PixelFormatTraits<pixelFormat>::ChannelType;
        auto pGrayBitmap = std::static_pointer_cast<Bitmap<pixelFormat>>(_pBitmap);
        //IBitmap::Save(pGrayBitmap, GetPathToTestFile("gray.pgm"));
        auto w = pGrayBitmap->GetWidth();
        auto h = pGrayBitmap->GetHeight();

        auto data = std::vector<ChannelType>(roi.width * roi.height * ChannelCount(pixelFormat));
        for (uint32_t i = 0; i < roi.height; ++i)
        {
            std::memcpy(&data[i * roi.width], pGrayBitmap->GetScanline(i + roi.y) + roi.x, roi.width * BytesPerPixel(pixelFormat));
        }

        auto median = data.begin() + data.size() / 2;
        std::nth_element(data.begin(), median, data.end());

        double thresholdPercent = 50;
        while (res->stars.size() < 40 && thresholdPercent > 8.0)
        {
            res->stars.clear();
            auto threshold = static_cast<ChannelType>(std::min(static_cast<uint32_t>(*median * (1 + thresholdPercent / 100)), static_cast<uint32_t>(std::numeric_limits<ChannelType>::max())));

            auto pData = pGrayBitmap->GetScanline(0);


            for (uint32_t i = roi.y; i < roi.y + roi.height; ++i)
            {
                for (uint32_t j = roi.x; j < roi.x + roi.width; ++j)
                {
                    if (!_visitedPixels[i * w + j] && pData[i * w + j] > threshold)
                    {
                        Star star{ Rect {static_cast<int32_t>(j), static_cast<int32_t>(i), 1, 1}, 0, 0, 0 };
                        InspectStar(star, threshold, pData, j, i, w, h, roi);
                        if (star.rect.width >= _minStarSize && star.rect.width <= _maxStarSize && star.rect.height >= _minStarSize && star.rect.height <= _maxStarSize)
                        {
                            star.center.x /= star.luminance;
                            star.center.y /= star.luminance;
                            res->stars.push_back(star);
                        }
                    }
                }
            }

            thresholdPercent /= res->stars.size() > 0 ? std::sqrt(40.0 / res->stars.size()) : 5;
        }

        //IBitmap::Save(pGrayBitmap, GetPathToTestFile("mask.pgm"));
        return res;
    }

    template <typename ChannelType>
    void InspectStar(Star& star, ChannelType threshold, ChannelType* pData, uint32_t x, uint32_t y, uint32_t w, uint32_t h, Rect roi)
    {
        ++star.pixelCount;        
        auto pixelLuminance = pData[y * w + x] - threshold;
        star.luminance += pixelLuminance;
        star.center.x += x * pixelLuminance;
        star.center.y += y * pixelLuminance;
        _visitedPixels[y * w + x] = true;

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

#endif
