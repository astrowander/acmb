#include "ppmencoder.h"
#include "Core/bitmap.h"

PpmEncoder::PpmEncoder(PpmMode ppmMode)
: _ppmMode(ppmMode)
{

}

void PpmEncoder::WriteBitmap(std::shared_ptr<IBitmap> pBitmap)
{
    if (!pBitmap)
        throw std::invalid_argument("pBitmap is null");

    auto pixelFormat = pBitmap->GetPixelFormat();

    switch (GetColorSpace(pixelFormat))
    {
    case ColorSpace::RGB:
        (_ppmMode == PpmMode::Binary) ? *_pStream << "P6" : *_pStream << "P3";
        break;
    case ColorSpace::Gray:
        (_ppmMode == PpmMode::Binary) ? *_pStream << "P5" : *_pStream << "P2";
        break;
    }

    *_pStream << std::endl;

    *_pStream << pBitmap->GetWidth() << " " << pBitmap->GetHeight() << std::endl;
    *_pStream << ((BytesPerChannel(pixelFormat) == 1) ? 255 : 65535);

    (_ppmMode == PpmMode::Binary) ? WriteBinary(pBitmap) : WriteText(pBitmap);
}

void PpmEncoder::WriteBinary(std::shared_ptr<IBitmap> pBitmap)
{
    *_pStream << std::endl;
    for (uint32_t i = 0; i < pBitmap->GetHeight(); ++i)
    {
        _pStream->write(pBitmap->GetPlanarScanline(i), pBitmap->GetWidth() * BytesPerPixel(pBitmap->GetPixelFormat()));
    }
}

void PpmEncoder::WriteText(std::shared_ptr<IBitmap> pBitmap)
{
    for (uint32_t i = 0; i < pBitmap->GetHeight(); ++i)
    {
        *_pStream << std::endl;
        for (uint32_t j = 0; j < pBitmap->GetWidth(); ++j)
        {
            for (uint32_t k = 0; k < ChannelCount(pBitmap->GetPixelFormat()); ++k)
                *_pStream << pBitmap->GetChannel(i, j, k) << " ";
        }
    }
}


