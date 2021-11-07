#include "registrator.h"
#include "./../Transforms/converter.h"

Registrator::Registrator(double threshold, uint32_t minStarSize, uint32_t maxStarSize, uint32_t hTiles, uint32_t vTiles)
: _threshold(threshold)
, _minStarSize(minStarSize)
, _maxStarSize(maxStarSize)
, _hTiles(hTiles)
, _vTiles(vTiles)
{

}

std::shared_ptr<DatasetTiles> Registrator::Registrate(std::shared_ptr<IBitmap> pBitmap)
{
    if (GetColorSpace(pBitmap->GetPixelFormat()) == ColorSpace::Gray)
    {
        _pBitmap = pBitmap;
    }
    else
    {
        _pBitmap = Convert(pBitmap, BytesPerChannel(pBitmap->GetPixelFormat()) == 1 ? PixelFormat::Gray8 : PixelFormat::Gray16);
    }   
    
    auto pDatasetTiles = std::make_shared<DatasetTiles>();

    const auto w = pBitmap->GetWidth();
    const auto h = pBitmap->GetHeight();
    const auto tileWidth = w / _hTiles;
    const auto tileHeight = h / _vTiles;

    for (uint32_t i = 0; i < _vTiles; ++i)
    {
        auto y = i * tileHeight;
        auto height = ((i == _vTiles - 1) ? h - y : tileHeight);

        for (uint32_t j = 0; j < _hTiles; ++j)
        {
            auto x = j * tileWidth;            
            auto width = ((j == _hTiles - 1) ? w - x : tileWidth);
            
            Rect roi{ (int32_t)x, (int32_t)y, width, height };

            std::shared_ptr<AlignmentDataset> dataset;
            if (BytesPerChannel(pBitmap->GetPixelFormat()) == 1)
                dataset = Registrate<PixelFormat::Gray8>(roi);
            else
                dataset = Registrate<PixelFormat::Gray16>(roi);

            if (!dataset->stars.empty())
            {
                std::sort(dataset->stars.begin(), dataset->stars.end(), [](auto& a, auto& b) {return a.luminance > b.luminance; });
                auto maxLuminance = dataset->stars[0].luminance;

                for (auto& star : dataset->stars)
                {
                    star.luminance /= maxLuminance;
                }

                dataset->starCount = dataset->stars.size();
                //const Star upperVal{ Rect {}, PointF{}, 0.5, 0 };
                //auto brightCount = std::upper_bound(dataset->stars.begin(), dataset->stars.end(), upperVal, [](const Star& a, const Star& b) { return a.luminance > b.luminance; }) - std::begin(dataset->stars);
                //brightCount = std::max((int)brightCount, 30);
                auto brightCount = 40;
                if (dataset->stars.size() > brightCount)
                {
                    dataset->stars.erase(std::begin(dataset->stars) + brightCount, std::end(dataset->stars));
                }

                std::cout << "tile " << i * _hTiles + j << " registered" << std::endl;
                std::cout << dataset->starCount << "stars are found" << std::endl;
                std::cout << "including " << dataset->stars.size() << " bright stars" << std::endl << std::endl;
            }

            pDatasetTiles->datasets.push_back(dataset);
            pDatasetTiles->totalStarCount += dataset->starCount;
        }
    }  

    return pDatasetTiles;
}
