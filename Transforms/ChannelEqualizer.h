#pragma once
#include "basetransform.h"
#include <array>
#include <functional>

class BaseChannelEqualizer : public BaseTransform
{
protected:
	BaseChannelEqualizer(IBitmapPtr pSrcBitmap);

public:
	static std::shared_ptr<BaseChannelEqualizer> Create(IBitmapPtr pSrcBitmap, const std::vector< std::function<uint32_t( uint32_t )>>& channelTransforms );
	static IBitmapPtr AutoEqualize( IBitmapPtr pSrcBitmap );
    virtual ~BaseChannelEqualizer() = default;
};

template<PixelFormat pixelFormat>
class ChannelEqualizer final: public BaseChannelEqualizer
{
	using ChannelType = typename PixelFormatTraits<pixelFormat>::ChannelType;
	static const uint32_t channelCount = PixelFormatTraits<pixelFormat>::channelCount;

	std::array<std::function<ChannelType( ChannelType )>, channelCount> _channelTransforms;

public:
	ChannelEqualizer( IBitmapPtr pSrcBitmap, const std::array< std::function<ChannelType( ChannelType )>, channelCount>& channelTransforms );
	void Run() override;
};
