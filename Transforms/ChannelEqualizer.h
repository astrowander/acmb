#pragma once
#include "basetransform.h"
#include "../Core/IParallel.h"
#include <array>

class BaseChannelEqualizer : public BaseTransform, public IParallel
{
protected:
	BaseChannelEqualizer(IBitmapPtr pSrcBitmap);

public:
	static std::shared_ptr<BaseChannelEqualizer> Create(IBitmapPtr pSrcBitmap, const std::vector<double>& multipliers);
	static IBitmapPtr AutoEqualize( IBitmapPtr pSrcBitmap );
};

template<PixelFormat pixelFormat>
class ChannelEqualizer : public BaseChannelEqualizer
{
	using ChannelType = typename PixelFormatTraits<pixelFormat>::ChannelType;
	static const uint32_t channelCount = PixelFormatTraits<pixelFormat>::channelCount;

	std::array<double, channelCount> _multipliers;

public:
	ChannelEqualizer(IBitmapPtr pSrcBitmap, const std::array<double, channelCount>& multipliers)
	: BaseChannelEqualizer(pSrcBitmap)
	, _multipliers(multipliers)
	{
	}

	void Run() override
	{
		_pDstBitmap = std::make_shared<Bitmap<pixelFormat>>(_pSrcBitmap->GetWidth(), _pSrcBitmap->GetHeight());
		DoParallelJobs();
	}

	void Job(uint32_t i) override
	{
		auto pSrcBitmap = std::static_pointer_cast<Bitmap<pixelFormat>>(_pSrcBitmap);
		auto pDstBitmap = std::static_pointer_cast<Bitmap<pixelFormat>>(_pDstBitmap);

		for (uint32_t ch = 0; ch < channelCount; ++ch)
		{
			auto pSrcScanline = pSrcBitmap->GetScanline(i) + ch;
			auto pDstScanline = pDstBitmap->GetScanline(i) + ch;

			for (uint32_t x = 0; x < pSrcBitmap->GetWidth(); ++x)
			{
				pDstScanline[0] = static_cast<ChannelType>(std::clamp<double>(pSrcScanline[0] * _multipliers[ch], 0, PixelFormatTraits<pixelFormat>::channelMax));
				pSrcScanline += channelCount;
				pDstScanline += channelCount;
			}
		}
	}
};


