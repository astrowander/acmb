#ifndef PPMENCODER_H
#define PPMENCODER_H

#include "enums.h"
#include "imageencoder.h"

class PpmEncoder : public ImageEncoder
{
    PpmMode _ppmMode;
public:
    PpmEncoder(PpmMode ppmMode);

    void WriteBitmap(std::shared_ptr<IBitmap> pBitmap) override;

private:
    void WriteBinary(std::shared_ptr<IBitmap> pBitmap);
    void WriteText(std::shared_ptr<IBitmap> pBitmap);
};

#endif // PPMENCODER_H
