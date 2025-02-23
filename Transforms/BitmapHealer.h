#pragma once
#include "basetransform.h"

ACMB_NAMESPACE_BEGIN

class BitmapHealer : public BaseTransform
{
public:

    struct Patch
    {
        Point from;
        Point to;
        int radius = 1;
        float gamma = 1.0f;
    };

    using Settings = std::vector<Patch>;

protected:

    BitmapHealer( IBitmapPtr pSrcBitmap, const Settings& settings );

    std::vector<Patch> _patches;

public:
    static std::shared_ptr<BitmapHealer> Create( IBitmapPtr pSrcBitmap, const Settings& settings );
    static std::shared_ptr<BitmapHealer> Create( PixelFormat, const Settings& settings );
    static IBitmapPtr ApplyTransform( IBitmapPtr pSrcBitmap, const Settings& settings );
};

ACMB_NAMESPACE_END
