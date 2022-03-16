#define _USE_MATH_DEFINES
#include <fstream>
#include "stacker.h"
#include "../Codecs/imagedecoder.h"
#include "FastAligner.h"
#include "../Geometry/delaunator.hpp"
#include "../Geometry/startrektransform.h"
#include "../Transforms/deaberratetransform.h"

Stacker::Stacker(std::vector<std::shared_ptr<ImageDecoder>> decoders, bool enableDeaberration)
: _gridWidth(0)
, _gridHeight(0)
, _enableDeaberration(enableDeaberration)
{
    for (auto pDecoder : decoders)
    {
        _stackingData.push_back({ pDecoder, {}, {} });
    }

    if (!decoders.empty())
    {
        _width = _stackingData[0].pDecoder->GetWidth();
        _height = _stackingData[0].pDecoder->GetHeight();
        _gridWidth = _width / gridSize + ((_width % gridSize) ? 1 : 0);
        _gridHeight = _height / gridSize + ((_height % gridSize) ? 1 : 0);
    }
}

void Stacker::Registrate(uint32_t hTileCount, uint32_t vTileCount, double threshold, uint32_t minStarSize, uint32_t maxStarSize)
{
    _hTileCount = hTileCount;
    _vTileCount = vTileCount;

    auto pRegistrator = std::make_unique<Registrator>(_hTileCount, _vTileCount, threshold, minStarSize, maxStarSize);
    for (auto& dsPair : _stackingData)
    {
        auto pBitmap = dsPair.pDecoder->ReadBitmap();
        
        if (_enableDeaberration)
        {
            auto pDeaberrateTransform = std::make_shared<DeaberrateTransform>(pBitmap, dsPair.pDecoder->GetCameraSettings());
            pBitmap = pDeaberrateTransform->RunAndGetBitmap();
        }
        
        pRegistrator->Registrate(pBitmap);
        dsPair.stars = pRegistrator->GetStars();

        for (const auto starVector : dsPair.stars)
        {
            dsPair.totalStarCount += starVector.size();
        }

        std::cout << dsPair.pDecoder->GetLastFileName() << " is registered" << std::endl;
        std::cout << dsPair.totalStarCount << " stars are found" << std::endl;
    }   

    //std::sort(std::begin(_stackingData), std::end(_stackingData), [](const auto& a, const auto& b) { return a.stars.size() > b.stars.size(); });
}

std::shared_ptr<IBitmap> Stacker::Stack(bool doAlignment)
{
    if (_stackingData.size() == 0)
        return nullptr;

    auto pRefBitmap = _stackingData[0].pDecoder->ReadBitmap();
    if (_enableDeaberration)
    {
        auto pDeaberrateTransform = std::make_shared<DeaberrateTransform>(pRefBitmap, _stackingData[0].pDecoder->GetCameraSettings());
        pRefBitmap = pDeaberrateTransform->RunAndGetBitmap();
    }

    if (_stackingData.size() == 1)
        return pRefBitmap;

    const auto& refStars = _stackingData[0].stars;

    std::vector<std::shared_ptr<FastAligner>> aligners;
    if (doAlignment)
    {
        for (const auto& refStarVector : refStars)
            aligners.push_back(std::make_shared<FastAligner>(refStarVector));
    }
    
    _stacked.resize(_width  * _height * ChannelCount(pRefBitmap->GetPixelFormat()));

    switch(pRefBitmap->GetPixelFormat())
    {
    case PixelFormat::Gray8:
        AddFirstBitmapToStack(std::static_pointer_cast<Bitmap<PixelFormat::Gray8>>(pRefBitmap));
        break;
    case PixelFormat::Gray16:
        AddFirstBitmapToStack(std::static_pointer_cast<Bitmap<PixelFormat::Gray16>>(pRefBitmap));
        break;
    case PixelFormat::RGB24:
        AddFirstBitmapToStack(std::static_pointer_cast<Bitmap<PixelFormat::RGB24>>(pRefBitmap));
        break;
    case PixelFormat::RGB48:
        AddFirstBitmapToStack(std::static_pointer_cast<Bitmap<PixelFormat::RGB48>>(pRefBitmap));
        break;
    default:
        throw std::runtime_error("pixel format should be known");
    }    

    for (uint32_t i = 1; i < _stackingData.size(); ++i)
    {
        auto pTargetBitmap = _stackingData[i].pDecoder->ReadBitmap();
        if (pRefBitmap->GetPixelFormat() != pTargetBitmap->GetPixelFormat())
            throw std::runtime_error("bitmaps in stack should have the same pixel format");

        if (_enableDeaberration)
        {
            auto pDeaberrateTransform = std::make_shared<DeaberrateTransform>(pTargetBitmap, _stackingData[0].pDecoder->GetCameraSettings());
            pTargetBitmap = pDeaberrateTransform->RunAndGetBitmap();
        }

        const auto& targetStars = _stackingData[i].stars;

        if (!doAlignment)
        {
            switch (pTargetBitmap->GetPixelFormat())
            {
            case PixelFormat::Gray8:
                AddBitmapToStackWithoutAlignment(std::static_pointer_cast<Bitmap<PixelFormat::Gray8>>(pTargetBitmap));
                break;
            case PixelFormat::Gray16:
                AddBitmapToStackWithoutAlignment(std::static_pointer_cast<Bitmap<PixelFormat::Gray16>>(pTargetBitmap));
                break;
            case PixelFormat::RGB24:
                AddBitmapToStackWithoutAlignment(std::static_pointer_cast<Bitmap<PixelFormat::RGB24>>(pTargetBitmap));
                break;
            case PixelFormat::RGB48:
                AddBitmapToStackWithoutAlignment(std::static_pointer_cast<Bitmap<PixelFormat::RGB48>>(pTargetBitmap));
                break;
            default:
                throw std::runtime_error("pixel format should be known");
            }

            continue;
        }

        std::cout << _stackingData[i].pDecoder->GetLastFileName() << " in process" << std::endl;

        MatchMap matches;
        
        for (uint32_t i = 0; i < refStars.size(); ++i)
        {
            aligners[i]->Align(targetStars[i]);
            auto tileMatches = aligners[i]->GetMatches();
            matches.insert(tileMatches.begin(), tileMatches.end());
        }

        std::cout << matches.size() << " matching stars" << std::endl;

        std::vector<double> coords;
        for (auto& match : matches)
        {
            coords.push_back(match.first.x);
            coords.push_back(match.first.y);
        }

        delaunator::Delaunator d(coords);  

        _grid.clear();
        _grid.resize(_gridWidth * _gridHeight);

        for (std::size_t i = 0; i < d.triangles.size(); i += 3)
        {
            Triangle targetTriangle{  PointF {d.coords[2 * d.triangles[i]], d.coords[2 * d.triangles[i] + 1]}, PointF {d.coords[2 * d.triangles[i + 1]], d.coords[2 * d.triangles[i + 1] + 1]}, PointF {d.coords[2 * d.triangles[i + 2]], d.coords[2 * d.triangles[i + 2] + 1]}  };
            Triangle refTriangle{ matches[targetTriangle.vertices[0]], matches[targetTriangle.vertices[1]], matches[targetTriangle.vertices[2]] };
            
            TriangleTransformPair pair = { refTriangle, agg::trans_affine(reinterpret_cast<double*>(refTriangle.vertices.data()), reinterpret_cast<double*>(targetTriangle.vertices.data())) };

            for (size_t j = 0; j < _gridWidth * _gridHeight; ++j)
            {
                RectF cell =
                {
                    static_cast<double>((j % _gridWidth) * gridSize),
                    static_cast<double>((j / _gridWidth) * gridSize),
                    gridSize,
                    gridSize
                };

                if (refTriangle.GetBoundingBox().Overlaps(cell))
                {
                    _grid[j].push_back(pair);
                }
            }
        }

        switch (pTargetBitmap->GetPixelFormat())
        {
        case PixelFormat::Gray8:
            AddBitmapToStack(std::static_pointer_cast<Bitmap<PixelFormat::Gray8>>(pTargetBitmap));
            break;
        case PixelFormat::Gray16:
            AddBitmapToStack(std::static_pointer_cast<Bitmap<PixelFormat::Gray16>>(pTargetBitmap));
            break;
        case PixelFormat::RGB24:
            AddBitmapToStack(std::static_pointer_cast<Bitmap<PixelFormat::RGB24>>(pTargetBitmap));
            break;
        case PixelFormat::RGB48:
            AddBitmapToStack(std::static_pointer_cast<Bitmap<PixelFormat::RGB48>>(pTargetBitmap));
            break;
        default:
            throw std::runtime_error("pixel format should be known");
        }

        std::cout << _stackingData[i].pDecoder->GetLastFileName() << " is stacked" << std::endl << std::endl;
    }
    

    switch(pRefBitmap->GetPixelFormat())
    {
    case PixelFormat::Gray8:
        return GetStackedBitmap<PixelFormat::Gray8>();
    case PixelFormat::Gray16:
        return GetStackedBitmap<PixelFormat::Gray16>();
    case PixelFormat::RGB24:
        return GetStackedBitmap<PixelFormat::RGB24>();
    case PixelFormat::RGB48:
        return GetStackedBitmap<PixelFormat::RGB48>();
    default:
        throw std::runtime_error("pixel format should be known");
    }

    return nullptr;
}
