#pragma once
#include "../Core/bitmap.h"
#include "../Core/pipeline.h"

ACMB_NAMESPACE_BEGIN

class BaseTransform : public IPipelineElement
{
protected:
    IBitmapPtr _pSrcBitmap;
    IBitmapPtr _pDstBitmap;

public:
    BaseTransform(IBitmapPtr pSrcBitmap);
    virtual ~BaseTransform() = default;

    virtual void Run() = 0;
    IBitmapPtr RunAndGetBitmap();
    void SetSrcBitmap(IBitmapPtr pSrcBitmap);

    virtual IBitmapPtr ProcessBitmap( IBitmapPtr pSrcBitmap ) override
    {
        SetSrcBitmap( pSrcBitmap );
        return RunAndGetBitmap();
    }
};

ACMB_NAMESPACE_END
