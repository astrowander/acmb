#ifndef PPMENCODER_H
#define PPMENCODER_H

#include "../../Core/enums.h"
#include "../../Codecs/imageencoder.h"

class PpmEncoder : public ImageEncoder
{
    PpmMode _ppmMode;
public:
    PpmEncoder(PpmMode ppmMode);

    void WriteBitmap(std::shared_ptr<IBitmap> pBitmap) override;

    static std::unordered_set<std::string> GetExtensions()
    {
        return { ".ppm", ".pgm" };
    }
private:
    template<uint32_t bytes>
    void WriteBinary(std::shared_ptr<IBitmap> pBitmap);

    void WriteText(std::shared_ptr<IBitmap> pBitmap);

    ADD_EXTENSIONS
};

#endif // PPMENCODER_H
