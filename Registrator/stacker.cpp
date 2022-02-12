#include "stacker.h"
#include "../Codecs/imagedecoder.h"
#include "FastAligner.h"
#include "../Geometry/delaunator.hpp"

Stacker::Stacker(std::vector<std::shared_ptr<ImageDecoder>> decoders)
{
    for (auto pDecoder : decoders)
    {
        _decoderStarPairs.push_back({ pDecoder, {} });
    }
}

void Stacker::Registrate(double threshold, uint32_t minStarSize, uint32_t maxStarSize)
{
    auto pRegistrator = std::make_unique<Registrator>(threshold, minStarSize, maxStarSize);
    for (auto& dsPair : _decoderStarPairs)
    {
        auto pBitmap = dsPair.first->ReadBitmap();
        dsPair.second = pRegistrator->Registrate(pBitmap);
        std::cout << dsPair.first->GetLastFileName() << " is registered" << std::endl;
        std::cout << dsPair.second.size() << " stars are found" << std::endl;
    }   

    std::sort(std::begin(_decoderStarPairs), std::end(_decoderStarPairs), [](const auto& a, const auto& b) { return a.second.size() > b.second.size(); });
}

std::shared_ptr<IBitmap> Stacker::Stack(bool doAlignment)
{
    if (_decoderStarPairs.size() == 0)
        return nullptr;

    auto pRefBitmap = _decoderStarPairs[0].first->ReadBitmap();
   // IBitmap::Save(pRefBitmap, "E:/test.ppm");
;    _width = pRefBitmap->GetWidth();
    _height = pRefBitmap->GetHeight();

    if (_decoderStarPairs.size() == 1)
        return pRefBitmap;

    const auto& refStars = _decoderStarPairs[0].second;
    auto pAligner = doAlignment ? std::make_shared<FastAligner>(refStars) : nullptr;
    
    _stacked.resize(_width  * _height * ChannelCount(pRefBitmap->GetPixelFormat()));

    switch(pRefBitmap->GetPixelFormat())
    {
    case PixelFormat::Gray8:
        AddBitmapToStack(std::static_pointer_cast<Bitmap<PixelFormat::Gray8>>(pRefBitmap));
        break;
    case PixelFormat::Gray16:
        AddBitmapToStack(std::static_pointer_cast<Bitmap<PixelFormat::Gray16>>(pRefBitmap));
        break;
    case PixelFormat::RGB24:
        AddBitmapToStack(std::static_pointer_cast<Bitmap<PixelFormat::RGB24>>(pRefBitmap));
        break;
    case PixelFormat::RGB48:
        AddBitmapToStack(std::static_pointer_cast<Bitmap<PixelFormat::RGB48>>(pRefBitmap));
        break;
    default:
        throw std::runtime_error("pixel format should be known");
    }

    

    for (uint32_t i = 1; i < _decoderStarPairs.size(); ++i)
    {
        auto pTargetBitmap = _decoderStarPairs[i].first->ReadBitmap();
        //IBitmap::Save(pTargetBitmap, "E:/test2.ppm");
        std::vector<std::pair<Triangle, agg::trans_affine>> trianglePairs;
        const auto& targetStars = _decoderStarPairs[i].second;

        if (doAlignment)
        {
            pAligner->Align(_decoderStarPairs[i].second);
            std::cout << _decoderStarPairs[i].first->GetLastFileName() << " in process" << std::endl;
            auto matches = pAligner->GetMatches();

            std::vector<double> coords;
            std::vector<size_t> targetStarIndices;
            for (auto& match : matches)
            {
                coords.push_back(targetStars[match.first].center.x);
                coords.push_back(targetStars[match.first].center.y);
                targetStarIndices.push_back(match.first);
            }

            delaunator::Delaunator d(coords);

            for (std::size_t i = 0; i < d.triangles.size(); i += 3) 
            {
                Triangle refTriangle { refStars[matches.at(targetStarIndices[d.triangles[i]])].center, refStars[matches.at(targetStarIndices[d.triangles[i + 1]])].center , refStars[matches.at(targetStarIndices[d.triangles[i + 2]])].center };
                Triangle targetTriangle { targetStars[targetStarIndices[d.triangles[i]]].center, targetStars[targetStarIndices[d.triangles[i + 1]]].center, targetStars[targetStarIndices[d.triangles[i + 2]]].center };

                trianglePairs.push_back({ refTriangle, agg::trans_affine(reinterpret_cast<double*>(refTriangle.vertices.data()), reinterpret_cast<double*>(targetTriangle.vertices.data())) });
            }

        }

        if (pRefBitmap->GetPixelFormat() != pRefBitmap->GetPixelFormat())
            throw std::runtime_error("bitmaps in stack should have the same pixel format");

        switch(pTargetBitmap->GetPixelFormat())
        {
        case PixelFormat::Gray8:
            AddBitmapToStack(std::static_pointer_cast<Bitmap<PixelFormat::Gray8>>(pTargetBitmap), trianglePairs);
            break;
        case PixelFormat::Gray16:
            AddBitmapToStack(std::static_pointer_cast<Bitmap<PixelFormat::Gray16>>(pTargetBitmap), trianglePairs);
            break;
        case PixelFormat::RGB24:
            AddBitmapToStack(std::static_pointer_cast<Bitmap<PixelFormat::RGB24>>(pTargetBitmap), trianglePairs);
            break;
        case PixelFormat::RGB48:
            AddBitmapToStack(std::static_pointer_cast<Bitmap<PixelFormat::RGB48>>(pTargetBitmap), trianglePairs);
            break;
        default:
            throw std::runtime_error("pixel format should be known");
        }

        std::cout << _decoderStarPairs[i].first->GetLastFileName() << " is stacked" << std::endl << std::endl;
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
