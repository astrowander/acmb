#ifndef IMAGEDECODER_H
#define IMAGEDECODER_H

#include <string>
#include <istream>
#include <iostream>
#include <memory>
#include "Core/imageparams.h"
#include <sstream>

class IBitmap;

class ImageDecoder : public ImageParams
{
protected:
    std::shared_ptr<std::istream> _pStream;
    virtual std::unique_ptr<std::istringstream> ReadLine();

public:

    virtual void Attach(std::shared_ptr<std::istream> pStream);
    virtual void Attach(const std::string& fileName);
    virtual void Detach();
    virtual ~ImageDecoder() = default;

    virtual std::shared_ptr<IBitmap> ReadBitmap() = 0;
    virtual std::shared_ptr<IBitmap> ReadStripe(uint32_t stripeHeight) = 0;

    virtual uint32_t GetCurrentScanline() const = 0;

    static std::shared_ptr<ImageDecoder> Create(const std::string& fileName);


};

#endif // IMAGEDECODER_H
