#include "stacker.h"
#include "Codecs/imagedecoder.h"
#include "aligner.h"
#include "alignmentdataset.h"
#include "registrator.h"
#include "Geometry/rect.h"



Stacker::Stacker(std::vector<std::shared_ptr<ImageDecoder> > decoders)
: _decoders(decoders)
{

}

std::shared_ptr<IBitmap> Stacker::Stack()
{
    if (_decoders.size() == 0)
        return nullptr;

    auto pRefBitmap = _decoders[0]->ReadBitmap();

    if (_decoders.size() == 1)
        return pRefBitmap;

    auto pRefDataset = Registrator::Registrate(pRefBitmap, 40, 5, 25);

    for (uint32_t i = 1; i < _decoders.size(); ++i)
    {
        auto pTargetBitmap = _decoders[i]->ReadBitmap();
        auto pTargetDataset = Registrator::Registrate(pTargetBitmap, 40, 5, 25);

        Aligner::Align(pRefDataset, pTargetDataset);

        for (uint32_t y = 0; y < pRefBitmap->GetHeight(); ++y)
        for (uint32_t x = 0; x < pRefBitmap->GetWidth(); ++x)
        {
            PointF targetPoint {static_cast<float>(x), static_cast<float>(y)};
            pTargetDataset->transform.transform(&targetPoint.x, &targetPoint.y);
        }
    }

    return nullptr;
}
