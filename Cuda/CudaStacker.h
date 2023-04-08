#pragma once
#include "./../Registrator/BaseStacker.h"
#include "CudaBasic.h"

ACMB_CUDA_NAMESPACE_BEGIN

class Stacker : public BaseStacker
{
    DynamicArrayF _means;
    DynamicArrayF _devs;
    DynamicArrayU16  _counts;

public:
    Stacker( const std::vector<Pipeline>& pipelines, StackMode stackMode );

    virtual std::shared_ptr<IBitmap> RegistrateAndStack() override;
    /// stacks registered images
    virtual std::shared_ptr<IBitmap> Stack() override;
};

ACMB_CUDA_NAMESPACE_END


