#pragma once
#include "basetransform.h"
#include <array>
#include <functional>

ACMB_NAMESPACE_BEGIN

class ChannelEqualizer : public BaseTransform
{
protected:
	ChannelEqualizer(IBitmapPtr pSrcBitmap);

public:
	static std::shared_ptr<ChannelEqualizer> Create(IBitmapPtr pSrcBitmap, const std::vector< std::function<uint32_t( uint32_t )>>& channelTransforms );
	static IBitmapPtr AutoEqualize( IBitmapPtr pSrcBitmap );
};

ACMB_NAMESPACE_END