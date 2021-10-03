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

    virtual std::shared_ptr<IBitmap> GetBitmap() = 0;


};

#endif // IMAGEDECODER_H
