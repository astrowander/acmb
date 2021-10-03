#ifndef IMAGEDECODER_H
#define IMAGEDECODER_H

#include <string>
#include <istream>
#include <memory>
#include "imageparams.h"
#include <sstream>

class IBitmap;

class ImageDecoder : public ImageParams
{
protected:
    std::unique_ptr<std::istream> _pStream;
    virtual std::unique_ptr<std::istringstream> ReadLine();

public:

    virtual void Attach(std::unique_ptr<std::istream> pStream);
    virtual void Attach(const std::string& fileName);
    virtual std::unique_ptr<std::istream> Detach();
    virtual ~ImageDecoder() = default;

    virtual std::shared_ptr<IBitmap> GetBitmap() = 0;


};

#endif // IMAGEDECODER_H
