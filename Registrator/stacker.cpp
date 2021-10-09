#include "stacker.h"
#include "Codecs/imagedecoder.h"

Stacker::Stacker(const std::vector<std::string> &fileNames)
{
    for (auto& fileName : fileNames)
        _decoders.push_back(ImageDecoder::Create(fileName));
}
