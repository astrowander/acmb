#include "bitmap.h"
#include <fstream>
#include <filesystem>
#include "Codecs/imagedecoder.h"
#include "Codecs/imageencoder.h"

std::shared_ptr<IBitmap> IBitmap::Create(const std::string &fileName)
{

    auto pDecoder = ImageDecoder::Create(fileName);
    pDecoder->Attach(fileName);
    return pDecoder->ReadBitmap();
}

void IBitmap::Save(std::shared_ptr<IBitmap> pBitmap, const std::string &fileName)
{
    if (!pBitmap)
        throw std::invalid_argument("pBitmap is null");

    auto pEncoder = ImageEncoder::Create(fileName);
    pEncoder->Attach(fileName);
    return pEncoder->WriteBitmap(pBitmap);
}
