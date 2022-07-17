#define _USE_MATH_DEFINES
//#define ENABLE_DIAGNOSTIC_MESSAGES

#include "stacker.h"
#include "AddingBitmapHelper.h"
#include "AddingBitmapWithAlignmentHelper.h"
#include "AlignmentHelper.h"
#include "GeneratingResultHelper.h"
#include "../Codecs/imagedecoder.h"
#include "../Geometry/delaunator.hpp"
#include "../Transforms/deaberratetransform.h"

void Log(const std::string& message)
{
#ifdef ENABLE_DIAGNOSTIC_MESSAGES
    std::cout << message << std::endl;
#endif
}

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

        Log(dsPair.pDecoder->GetLastFileName() + " is registered");
        Log(dsPair.totalStarCount + " stars are found");
    }   

    //std::sort(std::begin(_stackingData), std::end(_stackingData), [](const auto& a, const auto& b) { return a.stars.size() > b.stars.size(); });
}

std::shared_ptr<IBitmap> Stacker::Stack(bool doAlignment)
{
    if (_stackingData.size() == 0)
        return nullptr;
    
    Log(_stackingData[0].pDecoder->GetLastFileName() + " in process");

    auto pRefBitmap = _stackingData[0].pDecoder->ReadBitmap();

    Log(_stackingData[0].pDecoder->GetLastFileName() + " bitmap is read");

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
    
    _means.resize(_width  * _height * ChannelCount(pRefBitmap->GetPixelFormat()));
    _devs.resize(_width * _height * ChannelCount(pRefBitmap->GetPixelFormat()));
    _counts.resize(_width * _height * ChannelCount(pRefBitmap->GetPixelFormat()));

    CALL_HELPER(AddingBitmapHelper, pRefBitmap);

    Log(_stackingData[0].pDecoder->GetLastFileName() + " bitmap is stacked");

    for (uint32_t i = 1; i < _stackingData.size(); ++i)
    {
        Log(_stackingData[i].pDecoder->GetLastFileName() + " in process");
        auto pTargetBitmap = _stackingData[i].pDecoder->ReadBitmap();
        Log(_stackingData[i].pDecoder->GetLastFileName() + " bitmap is read");

        if (pRefBitmap->GetPixelFormat() != pTargetBitmap->GetPixelFormat())
            throw std::runtime_error("bitmaps in stack should have the same pixel format");

        if (_enableDeaberration)
        {
            auto pDeaberrateTransform = std::make_shared<DeaberrateTransform>(pTargetBitmap, _stackingData[0].pDecoder->GetCameraSettings());
            pTargetBitmap = pDeaberrateTransform->RunAndGetBitmap();
        }

        if (!doAlignment)
        {
            CALL_HELPER(AddingBitmapHelper, pTargetBitmap);
            continue;
        }

        
        StackWithAlignment(pRefBitmap, pTargetBitmap, i);
    }
    
    auto pRes = IBitmap::Create(_width, _height, pRefBitmap->GetPixelFormat());

    CALL_HELPER(GeneratingResultHelper, pRes);

    return pRes;
}

void Stacker::StackWithAlignment(IBitmapPtr pRefBitmap, IBitmapPtr pTargetBitmap, uint32_t i)
{
    _matches.clear();
    AlignmentHelper::Run(*this, i);
    Log(_matches.size() + " matching stars");

    std::vector<double> coords;
    for (auto& match : _matches)
    {
        coords.push_back(match.first.x);
        coords.push_back(match.first.y);
    }

    delaunator::Delaunator d(coords);

    Grid grid;
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

    Log(_stackingData[i].pDecoder->GetLastFileName() + " grid is calculated");

    CALL_HELPER(AddingBitmapWithAlignmentHelper, pTargetBitmap);

    Log(_stackingData[i].pDecoder->GetLastFileName() + " is stacked");
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

     _means.resize(_width * _height * ChannelCount(pRefBitmap->GetPixelFormat()));
     _devs.resize(_width * _height * ChannelCount(pRefBitmap->GetPixelFormat()));
     _counts.resize(_width * _height * ChannelCount(pRefBitmap->GetPixelFormat()));

    CALL_HELPER(AddingBitmapHelper, pRefBitmap);

    Log(_stackingData[0].pDecoder->GetLastFileName() + " bitmap is stacked");

    for (uint32_t i = 1; i < _stackingData.size(); ++i)
    {
        Log(_stackingData[i].pDecoder->GetLastFileName() + " in process");
        auto pTargetBitmap = _stackingData[i].pDecoder->ReadBitmap();
        Log(_stackingData[i].pDecoder->GetLastFileName() + " bitmap is read");

        if (pRefBitmap->GetPixelFormat() != pTargetBitmap->GetPixelFormat())
            throw std::runtime_error("bitmaps in stack should have the same pixel format");

        if (_enableDeaberration)
        {
            auto pDeaberrateTransform = std::make_shared<DeaberrateTransform>(pTargetBitmap, _stackingData[0].pDecoder->GetCameraSettings());
            pTargetBitmap = pDeaberrateTransform->RunAndGetBitmap();
        }

        pRegistrator->Registrate(pTargetBitmap);
        _stackingData[i].stars = pRegistrator->GetStars();

        StackWithAlignment(pRefBitmap, pTargetBitmap, i);
    }

    auto pRes = IBitmap::Create(_width, _height, pRefBitmap->GetPixelFormat());
    CALL_HELPER(GeneratingResultHelper, pRes);    

    return pRes;    
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