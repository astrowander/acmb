#pragma once
#include "./../Registrator/BaseStacker.h"
#include "AddBitmapWithAlignment.h"
#include <variant>

ACMB_NAMESPACE_BEGIN
class IBitmap;
ACMB_NAMESPACE_END

ACMB_CUDA_NAMESPACE_BEGIN

struct StackData;

class Stacker : public BaseStacker
{
    std::shared_ptr<StackData> _stackData;

    std::variant< AddBitmapWithAlignmentHelperU8, AddBitmapWithAlignmentHelperU16> _helper;

    virtual void CallAddBitmapHelper( std::shared_ptr<IBitmap> pBitmap ) override;
    virtual void CallAddBitmapWithAlignmentHelper( std::shared_ptr<IBitmap> pBitmap ) override;
    virtual std::shared_ptr<IBitmap> CallGeneratingResultHelper() override;

    void Init();

public:
    Stacker( const ImageParams& imageParams, StackMode stackMode );
};

ACMB_CUDA_NAMESPACE_END


