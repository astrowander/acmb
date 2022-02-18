#include "stacker.h"
#include "../Codecs/imagedecoder.h"
#include "FastAligner.h"
#include "../Geometry/delaunator.hpp"
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

void Stacker::Registrate(double threshold, uint32_t minStarSize, uint32_t maxStarSize)
{
    auto pRegistrator = std::make_unique<Registrator>(threshold, minStarSize, maxStarSize);
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
        dsPair.centralStars = pRegistrator->GetCentralStars();

        std::cout << dsPair.pDecoder->GetLastFileName() << " is registered" << std::endl;
        std::cout << dsPair.stars.size() << " stars are found" << std::endl;
        std::cout << dsPair.centralStars.size() << "central stars are found" << std::endl;
    }   

    std::sort(std::begin(_stackingData), std::end(_stackingData), [](const auto& a, const auto& b) { return a.stars.size() > b.stars.size(); });
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
    const auto& refCentralStars = _stackingData[0].centralStars;

    auto pAligner = doAlignment ? std::make_shared<FastAligner>(refStars) : nullptr;
    auto pCentralAligner = doAlignment ? std::make_shared<FastAligner>(refCentralStars) : nullptr;
    
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

        std::vector<std::pair<Triangle, agg::trans_affine>> trianglePairs;
        const auto& targetStars = _stackingData[i].stars;
        const auto& targetCentralStars = _stackingData[i].centralStars;

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
        pCentralAligner->Align(_stackingData[i].centralStars, _alignmentError);
        PointF centralPoint{ _width / 2.0, _height / 2.0 };
        PointF pSrc;
        PointF pDst;

        auto minDist = std::numeric_limits<double>::max();
        auto minPair = *std::begin(pCentralAligner->GetMatches());
        for (const auto& pair : pCentralAligner->GetMatches())
        {
            auto dist = centralPoint.Distance(refCentralStars[pair.second].center);
            if (dist < minDist)
            {
                minDist = dist;
                pSrc = refCentralStars[pair.second].center;
                pDst = targetCentralStars[pair.first].center;
            }
        }

        auto shift = pSrc.Distance(pDst);
        /*pCentralAligner->GetTransform().transform(&centralPointTransformed.x, &centralPointTransformed.y);
        auto dx = centralPointTransformed.x - centralPoint.x;
        auto dy = centralPointTransformed.y - centralPoint.y;

        auto shift = sqrt(dx * dx + dy * dy);*/

        pAligner->Align(_stackingData[i].stars, pCentralAligner->GetTransform(), _alignmentError);

        auto matches = pAligner->GetMatches();
        std::cout << matches.size() << " matching stars" << std::endl;

        std::vector<double> coords;
        std::vector<size_t> targetStarIndices;
        for (auto& match : matches)
        {
            coords.push_back(targetStars[match.first].center.x);
            coords.push_back(targetStars[match.first].center.y);
            targetStarIndices.push_back(match.first);
        }

        delaunator::Delaunator d(coords);  

        _grid.clear();
        _grid.resize(_gridWidth * _gridHeight);

        for (std::size_t i = 0; i < d.triangles.size(); i += 3)
        {
            Triangle refTriangle{ refStars[matches.at(targetStarIndices[d.triangles[i]])].center, refStars[matches.at(targetStarIndices[d.triangles[i + 1]])].center , refStars[matches.at(targetStarIndices[d.triangles[i + 2]])].center };
            Triangle targetTriangle{ targetStars[targetStarIndices[d.triangles[i]]].center, targetStars[targetStarIndices[d.triangles[i + 1]]].center, targetStars[targetStarIndices[d.triangles[i + 2]]].center };

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
