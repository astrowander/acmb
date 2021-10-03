#include "bitmap.h"
#include <fstream>
#include <filesystem>
#include "Codecs/PPM/ppmdecoder.h"
#include "Codecs/PPM/ppmencoder.h"

std::shared_ptr<IBitmap> IBitmap::Create(const std::string &fileName)
{
    auto path = std::filesystem::path(fileName);
    auto extension = path.extension();
    std::unique_ptr<ImageDecoder> pDecoder;
    if (extension == ".pgm" || extension == ".ppm")
    {
        pDecoder.reset(new PpmDecoder());
    }

    if (!pDecoder)
        throw std::invalid_argument("fileName");

    pDecoder->Attach(fileName);
    return pDecoder->GetBitmap();
}

void IBitmap::Save(std::shared_ptr<IBitmap> pBitmap, const std::string &fileName)
{
    if (!pBitmap)
        throw std::invalid_argument("pBitmap is null");

    auto path = std::filesystem::path(fileName);
    auto extension = path.extension();
    std::unique_ptr<ImageEncoder> pEncoder;
    if (extension == ".pgm" || extension == ".ppm")
    {
        pEncoder.reset(new PpmEncoder(PpmMode::Binary));
    }

    if (!pEncoder)
        throw std::invalid_argument("fileName");

    pEncoder->Attach(fileName);
    return pEncoder->WriteBitmap(pBitmap);
}

