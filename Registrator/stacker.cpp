#include "stacker.h"
#include "Codecs/imagedecoder.h"
#include "aligner.h"
#include "alignmentdataset.h"
#include "registrator.h"




Stacker::Stacker(std::vector<std::shared_ptr<ImageDecoder> > decoders)
: _decoders(decoders)
{

}

std::shared_ptr<IBitmap> Stacker::Stack()
{
    if (_decoders.size() == 0)
        return nullptr;

    auto pRefBitmap = _decoders[0]->ReadBitmap();
    _width = pRefBitmap->GetWidth();
    _height = pRefBitmap->GetHeight();

    if (_decoders.size() == 1)
        return pRefBitmap;

    auto pRefDataset = Registrator::Registrate(pRefBitmap, 40, 5, 25);
    _stacked.resize(_width  * _height * ChannelCount(pRefBitmap->GetPixelFormat()));

    switch(pRefBitmap->GetPixelFormat())
    {
    case PixelFormat::Gray8:
        AddBitmapToStack(std::static_pointer_cast<Bitmap<PixelFormat::Gray8>>(pRefBitmap), 0, agg::trans_affine());
        break;
    case PixelFormat::Gray16:
        AddBitmapToStack(std::static_pointer_cast<Bitmap<PixelFormat::Gray16>>(pRefBitmap), 0, agg::trans_affine());
        break;
    case PixelFormat::RGB24:
        AddBitmapToStack(std::static_pointer_cast<Bitmap<PixelFormat::RGB24>>(pRefBitmap), 0, agg::trans_affine());
        break;
    case PixelFormat::RGB48:
        AddBitmapToStack(std::static_pointer_cast<Bitmap<PixelFormat::RGB48>>(pRefBitmap), 0, agg::trans_affine());
        break;
    default:
        throw std::runtime_error("pixel format should be known");
    }

    for (uint32_t i = 1; i < _decoders.size(); ++i)
    {
        auto pTargetBitmap = _decoders[i]->ReadBitmap();
        auto pTargetDataset = Registrator::Registrate(pTargetBitmap, 40, 5, 25);

        Aligner::Align(pRefDataset, pTargetDataset);

        if (pRefBitmap->GetPixelFormat() != pRefBitmap->GetPixelFormat())
            throw std::runtime_error("bitmaps in stack should have the same pixel format");

        switch(pTargetBitmap->GetPixelFormat())
        {
        case PixelFormat::Gray8:
            AddBitmapToStack(std::static_pointer_cast<Bitmap<PixelFormat::Gray8>>(pTargetBitmap), i, pTargetDataset->transform);
            break;
        case PixelFormat::Gray16:
            AddBitmapToStack(std::static_pointer_cast<Bitmap<PixelFormat::Gray16>>(pTargetBitmap), i, pTargetDataset->transform);
            break;
        case PixelFormat::RGB24:
            AddBitmapToStack(std::static_pointer_cast<Bitmap<PixelFormat::RGB24>>(pTargetBitmap), i, pTargetDataset->transform);
            break;
        case PixelFormat::RGB48:
            AddBitmapToStack(std::static_pointer_cast<Bitmap<PixelFormat::RGB48>>(pTargetBitmap), i, pTargetDataset->transform);
            break;
        default:
            throw std::runtime_error("pixel format should be known");
        }
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
