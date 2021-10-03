#ifndef IMAGEENCODER_H
#define IMAGEENCODER_H

#include <string>
#include <ostream>
#include <memory>

class IBitmap;

class ImageEncoder
{
protected:
    std::shared_ptr<std::ostream> _pStream;

public:

    virtual void Attach(std::shared_ptr<std::ostream> pStream);
    virtual void Attach(const std::string& fileName);
    virtual void Detach();

    virtual ~ImageEncoder() = default;

    virtual void WriteBitmap(std::shared_ptr<IBitmap> pBitmap) = 0;
};

#endif // IMAGEENCODER_H
