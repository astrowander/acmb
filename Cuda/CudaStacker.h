#pragma once
#include "./../Registrator/BaseStacker.h"
#include "CudaBasic.h"
#include "AddBitmapWithAlignment.h"
#include <variant>

ACMB_CUDA_NAMESPACE_BEGIN

class Stacker : public BaseStacker
{
    DynamicArrayF _means;
    DynamicArrayF _devs;
    DynamicArrayU16 _counts;

    std::variant<DynamicArrayU8, DynamicArrayU16> _cudaBitmap;
    std::variant< AddBitmapWithAlignmentHelperU8, AddBitmapWithAlignmentHelperU16> _helper;

    virtual void CallAddBitmapHelper( IBitmapPtr pBitmap ) override;
    virtual void CallAddBitmapWithAlignmentHelper( IBitmapPtr pBitmap ) override;
    virtual IBitmapPtr CallGeneratingResultHelper() override;

    void Init();

public:
    Stacker( const std::vector<Pipeline>& pipelines, StackMode stackMode );
    Stacker( const ImageParams& imageParams, StackMode stackMode );
};

ACMB_CUDA_NAMESPACE_END


