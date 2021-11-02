#include "stacker.h"
#include "../Codecs/imagedecoder.h"
#include "aligner.h"
#include "alignmentdataset.h"
#include "registrator.h"

Stacker::Stacker(std::vector<std::shared_ptr<ImageDecoder>> decoders)
{
    for (auto& pDecoder : decoders)
    {
        _decoderDatasetPairs.push_back({ pDecoder, nullptr });
    }
}

void Stacker::Registrate(double threshold, uint32_t minStarSize, uint32_t maxStarSize)
{
    auto pRegistrator = std::make_unique<Registrator>(threshold, minStarSize, maxStarSize);
    for (auto& decoderDatasetPair : _decoderDatasetPairs)
    {
        auto pBitmap = decoderDatasetPair.first->ReadBitmap();
        decoderDatasetPair.second = pRegistrator->Registrate(pBitmap);
        std::cout << decoderDatasetPair.first->GetLastFileName() << " is registered" << std::endl;
        std::cout << decoderDatasetPair.second->starCount << " stars are found" << std::endl;
        std::cout << decoderDatasetPair.second->stars.size() << " bright stars are found" << std::endl << std::endl;
    }

    std::sort(std::begin(_decoderDatasetPairs), std::end(_decoderDatasetPairs), [](const auto& a, const auto& b) { return a.second->starCount > b.second->starCount; });
}

std::shared_ptr<IBitmap> Stacker::Stack(bool doAlignment)
{
    if (_decoderDatasetPairs.size() == 0)
        return nullptr;

    auto pRefBitmap = _decoderDatasetPairs[0].first->ReadBitmap();
    IBitmap::Save(pRefBitmap, "E:/test.ppm");
;    _width = pRefBitmap->GetWidth();
    _height = pRefBitmap->GetHeight();

    if (_decoderDatasetPairs.size() == 1)
        return pRefBitmap;

    auto pRefDataset = doAlignment ? _decoderDatasetPairs[0].second : nullptr;
    auto pAligner = doAlignment ? std::make_shared<Aligner>(pRefDataset) : nullptr;

    _stacked.resize(_width  * _height * ChannelCount(pRefBitmap->GetPixelFormat()));

    switch(pRefBitmap->GetPixelFormat())
    {
    case PixelFormat::Gray8:
        AddBitmapToStack(std::static_pointer_cast<Bitmap<PixelFormat::Gray8>>(pRefBitmap), agg::trans_affine());
        break;
    case PixelFormat::Gray16:
        AddBitmapToStack(std::static_pointer_cast<Bitmap<PixelFormat::Gray16>>(pRefBitmap), agg::trans_affine());
        break;
    case PixelFormat::RGB24:
        AddBitmapToStack(std::static_pointer_cast<Bitmap<PixelFormat::RGB24>>(pRefBitmap), agg::trans_affine());
        break;
    case PixelFormat::RGB48:
        AddBitmapToStack(std::static_pointer_cast<Bitmap<PixelFormat::RGB48>>(pRefBitmap), agg::trans_affine());
        break;
    default:
        throw std::runtime_error("pixel format should be known");
    }

    

    for (uint32_t i = 1; i < _decoderDatasetPairs.size(); ++i)
    {
        auto pTargetBitmap = _decoderDatasetPairs[i].first->ReadBitmap();
        IBitmap::Save(pTargetBitmap, "E:/test2.ppm");
        auto pTargetDataset = doAlignment ? _decoderDatasetPairs[i].second : nullptr;

        if (doAlignment)
        {
            std::cout << _decoderDatasetPairs[i].first->GetLastFileName() << " in process" << std::endl;
            pAligner->Align(pTargetDataset);
            std::cout << _decoderDatasetPairs[i].first->GetLastFileName() << " is aligned" << std::endl;
            std::cout << "tx = " << _decoderDatasetPairs[i].second->transform.tx << ", ty = " << _decoderDatasetPairs[i].second->transform.ty
                << ", rotation = " << _decoderDatasetPairs[i].second->transform.rotation() * 180 / 3.1416 << std::endl;
        }

        if (pRefBitmap->GetPixelFormat() != pRefBitmap->GetPixelFormat())
            throw std::runtime_error("bitmaps in stack should have the same pixel format");

        switch(pTargetBitmap->GetPixelFormat())
        {
        case PixelFormat::Gray8:
            AddBitmapToStack(std::static_pointer_cast<Bitmap<PixelFormat::Gray8>>(pTargetBitmap), doAlignment ? pTargetDataset->transform : agg::trans_affine());
            break;
        case PixelFormat::Gray16:
            AddBitmapToStack(std::static_pointer_cast<Bitmap<PixelFormat::Gray16>>(pTargetBitmap), doAlignment ? pTargetDataset->transform : agg::trans_affine());
            break;
        case PixelFormat::RGB24:
            AddBitmapToStack(std::static_pointer_cast<Bitmap<PixelFormat::RGB24>>(pTargetBitmap), doAlignment ? pTargetDataset->transform : agg::trans_affine());
            break;
        case PixelFormat::RGB48:
            AddBitmapToStack(std::static_pointer_cast<Bitmap<PixelFormat::RGB48>>(pTargetBitmap), doAlignment ? pTargetDataset->transform : agg::trans_affine());
            break;
        default:
            throw std::runtime_error("pixel format should be known");
        }

        std::cout << _decoderDatasetPairs[i].first->GetLastFileName() << " is stacked" << std::endl << std::endl;
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
