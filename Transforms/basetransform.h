#pragma once
#include "../Core/bitmap.h"
#include "../Core/pipeline.h"

ACMB_NAMESPACE_BEGIN
/// <summary>
/// Abstract class for an image transformation
/// </summary>
class BaseTransform : public IPipelineElement
{
protected:
    IBitmapPtr _pSrcBitmap;
    IBitmapPtr _pDstBitmap;

    /// this function is called in SetSrcBitmap. It checks if the given bitmap is compatible with the given settings. If not throw an exception
    virtual void ValidateSettings() = 0;

public:
    BaseTransform(IBitmapPtr pSrcBitmap);
    virtual ~BaseTransform() = default;

    /// Override this in the derived class
    virtual void Run() = 0;
    /// Runs the transform and returns its result
    IBitmapPtr RunAndGetBitmap();

    /// Sets source image
    void SetSrcBitmap(IBitmapPtr pSrcBitmap);

    /// Used in pipelines. Receives an image from previous element and passes the result to the next one
    virtual IBitmapPtr ProcessBitmap( IBitmapPtr pSrcBitmap ) override
    {
        SetSrcBitmap( pSrcBitmap );
        auto res = RunAndGetBitmap();
        _pSrcBitmap.reset();
        _pDstBitmap.reset();
        return res;
    }
};

ACMB_NAMESPACE_END
