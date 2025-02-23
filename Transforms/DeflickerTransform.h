#pragma once

#include "basetransform.h"

ACMB_NAMESPACE_BEGIN

class DeflickerTransform
{
public:
    struct Settings
    {
        std::span<IBitmapPtr> bitmaps;
        int iterations = 1;
    };
protected:
    DeflickerTransform( const Settings& settings );
    Settings _settings;

public:
    virtual void Run() = 0;

    static std::shared_ptr<DeflickerTransform> Create( const Settings& settings );
    static void Deflicker( const Settings& settings );
};

ACMB_NAMESPACE_END
