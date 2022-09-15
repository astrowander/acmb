#ifndef CONVERTER_H
#define CONVERTER_H
#include "../Core/bitmap.h"
#include <stdexcept>
#include "basetransform.h"

class BaseConverter : public BaseTransform
{
protected:
    BaseConverter(IBitmapPtr pSrcBitmap);

public:
    static std::shared_ptr<BaseConverter> Create(IBitmapPtr pSrcBitmap, PixelFormat dstPixelFormat);
    static IBitmapPtr Convert(IBitmapPtr pSrcBitmap, PixelFormat dstPixelFormat);
};

template <PixelFormat srcPixelFormat, PixelFormat dstPixelFormat>
class Converter final: public BaseConverter
{
public:
    Converter( IBitmapPtr pSrcBitmap );
    void Run() override;
};

#endif // CONVERTER_H


