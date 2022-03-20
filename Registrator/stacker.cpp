#define _USE_MATH_DEFINES
#include <fstream>
#include "stacker.h"
#include "AddingBitmapHelper.h"
#include "AddingBitmapWithAlignmentHelper.h"
#include "AlignmentHelper.h"
#include "GeneratingResultHelper.h"
#include "../Codecs/imagedecoder.h"
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
    
    std::cout << _stackingData[0].pDecoder->GetLastFileName() << " in process" << std::endl;
    auto pRefBitmap = _stackingData[0].pDecoder->ReadBitmap();
    std::cout << _stackingData[0].pDecoder->GetLastFileName() << " bitmap is read" << std::endl;

    if (_enableDeaberration)
    {
        auto pDeaberrateTransform = std::make_shared<DeaberrateTransform>(pRefBitmap, _stackingData[0].pDecoder->GetCameraSettings());
        pRefBitmap = pDeaberrateTransform->RunAndGetBitmap();
    }

    if (_stackingData.size() == 1)
        return pRefBitmap;

    const auto& refStars = _stackingData[0].stars;

    if (doAlignment)
    {
        _aligners.clear();

        for (const auto& refStarVector : refStars)
            _aligners.push_back(std::make_shared<FastAligner>(refStarVector));
    }
    
    _stacked.resize(_width  * _height * ChannelCount(pRefBitmap->GetPixelFormat()));

    switch(pRefBitmap->GetPixelFormat())
    {
    case PixelFormat::Gray8:
        AddingBitmapHelper<PixelFormat::Gray8>::AddBitmap(*this, std::static_pointer_cast<Bitmap<PixelFormat::Gray8>>(pRefBitmap));
        break;
    case PixelFormat::Gray16:
        AddingBitmapHelper<PixelFormat::Gray16>::AddBitmap(*this, std::static_pointer_cast<Bitmap<PixelFormat::Gray16>>(pRefBitmap));
        break;
    case PixelFormat::RGB24:
        AddingBitmapHelper<PixelFormat::RGB24>::AddBitmap(*this, std::static_pointer_cast<Bitmap<PixelFormat::RGB24>>(pRefBitmap));
        break;
    case PixelFormat::RGB48:
        AddingBitmapHelper<PixelFormat::RGB48>::AddBitmap(*this, std::static_pointer_cast<Bitmap<PixelFormat::RGB48>>(pRefBitmap));
        break;
    default:
        throw std::runtime_error("pixel format should be known");
    }    

    std::cout << _stackingData[0].pDecoder->GetLastFileName() << " bitmap is stacked" << std::endl;

    for (uint32_t i = 1; i < _stackingData.size(); ++i)
    {
        std::cout << _stackingData[i].pDecoder->GetLastFileName() << " in process" << std::endl;

        auto pTargetBitmap = _stackingData[i].pDecoder->ReadBitmap();

        std::cout << _stackingData[i].pDecoder->GetLastFileName() << " bitmap is read" << std::endl;
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
                AddingBitmapHelper<PixelFormat::Gray8>::AddBitmap(*this, std::static_pointer_cast<Bitmap<PixelFormat::Gray8>>(pRefBitmap));
                break;
            case PixelFormat::Gray16:
                AddingBitmapHelper<PixelFormat::Gray16>::AddBitmap(*this, std::static_pointer_cast<Bitmap<PixelFormat::Gray16>>(pRefBitmap));
                break;
            case PixelFormat::RGB24:
                AddingBitmapHelper<PixelFormat::RGB24>::AddBitmap(*this, std::static_pointer_cast<Bitmap<PixelFormat::RGB24>>(pRefBitmap));
                break;
            case PixelFormat::RGB48:
                AddingBitmapHelper<PixelFormat::RGB48>::AddBitmap(*this, std::static_pointer_cast<Bitmap<PixelFormat::RGB48>>(pRefBitmap));
                break;
            default:
                throw std::runtime_error("pixel format should be known");
            }

            continue;
        }

        
        _matches.clear();
        AlignmentHelper::Align(*this, i);

        std::cout << _matches.size() << " matching stars" << std::endl;

        std::vector<double> coords;
        for (auto& match : _matches)
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
            Triangle refTriangle{ _matches[targetTriangle.vertices[0]], _matches[targetTriangle.vertices[1]], _matches[targetTriangle.vertices[2]] };
            
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

        std::cout << _stackingData[i].pDecoder->GetLastFileName() << " grid is calculated" << std::endl;
        switch (pTargetBitmap->GetPixelFormat())
        {
        case PixelFormat::Gray8:
            AddingBitmapWithAlignmentHelper<PixelFormat::Gray8>::AddBitmap(*this, std::static_pointer_cast<Bitmap<PixelFormat::Gray8>>(pTargetBitmap));
            break;
        case PixelFormat::Gray16:
            AddingBitmapWithAlignmentHelper<PixelFormat::Gray16>::AddBitmap(*this, std::static_pointer_cast<Bitmap<PixelFormat::Gray16>>(pTargetBitmap));
            break;
        case PixelFormat::RGB24:
            AddingBitmapWithAlignmentHelper<PixelFormat::RGB24>::AddBitmap(*this, std::static_pointer_cast<Bitmap<PixelFormat::RGB24>>(pTargetBitmap));
            break;
        case PixelFormat::RGB48:
            AddingBitmapWithAlignmentHelper<PixelFormat::RGB48>::AddBitmap(*this, std::static_pointer_cast<Bitmap<PixelFormat::RGB48>>(pTargetBitmap));
            break;
        default:
            throw std::runtime_error("pixel format should be known");
        }

        std::cout << _stackingData[i].pDecoder->GetLastFileName() << " is stacked" << std::endl << std::endl;
    }
    

    switch(pRefBitmap->GetPixelFormat())
    {
    case PixelFormat::Gray8:
        return GeneratingResultHelper<PixelFormat::Gray8>::GenerateResult(*this);
    case PixelFormat::Gray16:
        return GeneratingResultHelper<PixelFormat::Gray16>::GenerateResult(*this);
    case PixelFormat::RGB24:
        return GeneratingResultHelper<PixelFormat::RGB24>::GenerateResult(*this);
    case PixelFormat::RGB48:
        return GeneratingResultHelper<PixelFormat::RGB48>::GenerateResult(*this);
    default:
        throw std::runtime_error("pixel format should be known");
    }

    return nullptr;
}

std::shared_ptr<IBitmap>  Stacker::RegistrateAndStack(uint32_t hTileCount, uint32_t vTileCount, double threshold, uint32_t minStarSize, uint32_t maxStarSize)
{
    if (_stackingData.size() == 0)
        return nullptr;   

    _hTileCount = hTileCount;
    _vTileCount = vTileCount;

    auto pRefBitmap = _stackingData[0].pDecoder->ReadBitmap();

    if (_enableDeaberration)
    {
        auto pDeaberrateTransform = std::make_shared<DeaberrateTransform>(pRefBitmap, _stackingData[0].pDecoder->GetCameraSettings());
        pRefBitmap = pDeaberrateTransform->RunAndGetBitmap();
    }

    if (_stackingData.size() == 1)
        return pRefBitmap;

    auto pRegistrator = std::make_unique<Registrator>(_hTileCount, _vTileCount, threshold, minStarSize, maxStarSize);

    pRegistrator->Registrate(pRefBitmap);
    _stackingData[0].stars = pRegistrator->GetStars();

    const auto& refStars = _stackingData[0].stars;

     _aligners.clear();
     for (const auto& refStarVector : refStars)
         _aligners.push_back(std::make_shared<FastAligner>(refStarVector));    

    _stacked.resize(_width * _height * ChannelCount(pRefBitmap->GetPixelFormat()));

    switch (pRefBitmap->GetPixelFormat())
    {
    case PixelFormat::Gray8:
        AddingBitmapHelper<PixelFormat::Gray8>::AddBitmap(*this, std::static_pointer_cast<Bitmap<PixelFormat::Gray8>>(pRefBitmap));
        break;
    case PixelFormat::Gray16:
        AddingBitmapHelper<PixelFormat::Gray16>::AddBitmap(*this, std::static_pointer_cast<Bitmap<PixelFormat::Gray16>>(pRefBitmap));
        break;
    case PixelFormat::RGB24:
        AddingBitmapHelper<PixelFormat::RGB24>::AddBitmap(*this, std::static_pointer_cast<Bitmap<PixelFormat::RGB24>>(pRefBitmap));
        break;
    case PixelFormat::RGB48:
        AddingBitmapHelper<PixelFormat::RGB48>::AddBitmap(*this, std::static_pointer_cast<Bitmap<PixelFormat::RGB48>>(pRefBitmap));
        break;
    default:
        throw std::runtime_error("pixel format should be known");
    }

    std::cout << _stackingData[0].pDecoder->GetLastFileName() << " bitmap is stacked" << std::endl;

    for (uint32_t i = 1; i < _stackingData.size(); ++i)
    {
        std::cout << _stackingData[i].pDecoder->GetLastFileName() << " in process" << std::endl;

        auto pTargetBitmap = _stackingData[i].pDecoder->ReadBitmap();

        std::cout << _stackingData[i].pDecoder->GetLastFileName() << " bitmap is read" << std::endl;

        if (pRefBitmap->GetPixelFormat() != pTargetBitmap->GetPixelFormat())
            throw std::runtime_error("bitmaps in stack should have the same pixel format");

        if (_enableDeaberration)
        {
            auto pDeaberrateTransform = std::make_shared<DeaberrateTransform>(pTargetBitmap, _stackingData[0].pDecoder->GetCameraSettings());
            pTargetBitmap = pDeaberrateTransform->RunAndGetBitmap();
        }

        pRegistrator->Registrate(pTargetBitmap);
        _stackingData[i].stars = pRegistrator->GetStars();
        const auto& targetStars = _stackingData[0].stars;

        _matches.clear();
        AlignmentHelper::Align(*this, i);

        std::cout << _matches.size() << " matching stars" << std::endl;

        std::vector<double> coords;
        for (auto& match : _matches)
        {
            coords.push_back(match.first.x);
            coords.push_back(match.first.y);
        }

        delaunator::Delaunator d(coords);

        _grid.clear();
        _grid.resize(_gridWidth * _gridHeight);

        for (std::size_t i = 0; i < d.triangles.size(); i += 3)
        {
            Triangle targetTriangle{ PointF {d.coords[2 * d.triangles[i]], d.coords[2 * d.triangles[i] + 1]}, PointF {d.coords[2 * d.triangles[i + 1]], d.coords[2 * d.triangles[i + 1] + 1]}, PointF {d.coords[2 * d.triangles[i + 2]], d.coords[2 * d.triangles[i + 2] + 1]} };
            Triangle refTriangle{ _matches[targetTriangle.vertices[0]], _matches[targetTriangle.vertices[1]], _matches[targetTriangle.vertices[2]] };

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

        std::cout << _stackingData[i].pDecoder->GetLastFileName() << " grid is calculated" << std::endl;
        switch (pTargetBitmap->GetPixelFormat())
        {
        case PixelFormat::Gray8:
            AddingBitmapWithAlignmentHelper<PixelFormat::Gray8>::AddBitmap(*this, std::static_pointer_cast<Bitmap<PixelFormat::Gray8>>(pTargetBitmap));
            break;
        case PixelFormat::Gray16:
            AddingBitmapWithAlignmentHelper<PixelFormat::Gray16>::AddBitmap(*this, std::static_pointer_cast<Bitmap<PixelFormat::Gray16>>(pTargetBitmap));
            break;
        case PixelFormat::RGB24:
            AddingBitmapWithAlignmentHelper<PixelFormat::RGB24>::AddBitmap(*this, std::static_pointer_cast<Bitmap<PixelFormat::RGB24>>(pTargetBitmap));
            break;
        case PixelFormat::RGB48:
            AddingBitmapWithAlignmentHelper<PixelFormat::RGB48>::AddBitmap(*this, std::static_pointer_cast<Bitmap<PixelFormat::RGB48>>(pTargetBitmap));
            break;
        default:
            throw std::runtime_error("pixel format should be known");
        }

        std::cout << _stackingData[i].pDecoder->GetLastFileName() << " is stacked" << std::endl << std::endl;
    }


    switch (pRefBitmap->GetPixelFormat())
    {
    case PixelFormat::Gray8:
        return GeneratingResultHelper<PixelFormat::Gray8>::GenerateResult(*this);
    case PixelFormat::Gray16:
        return GeneratingResultHelper<PixelFormat::Gray16>::GenerateResult(*this);
    case PixelFormat::RGB24:
        return GeneratingResultHelper<PixelFormat::RGB24>::GenerateResult(*this);
    case PixelFormat::RGB48:
        return GeneratingResultHelper<PixelFormat::RGB48>::GenerateResult(*this);
    default:
        throw std::runtime_error("pixel format should be known");
    }

    return nullptr;    
}

void Stacker::ChooseTriangle(PointF p, std::pair<Triangle, agg::trans_affine>& lastPair, const Stacker::GridCell& trianglePairs)
{
    if (lastPair.first.IsPointInside(p))
        return;

    double minDist = std::numeric_limits<double>::max();
    TriangleTransformPair nearest;

    for (const auto& pair : trianglePairs)
    {
        if (pair.first.IsPointInside(p))
        {
            lastPair = pair;
            return;
        }

        double dist = p.Distance(pair.first.GetCenter());
        if (dist < minDist)
        {
            nearest = pair;
            minDist = dist;
        }
    }

    lastPair = nearest;
}