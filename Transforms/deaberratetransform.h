#ifndef DEABERRATETRANSFORM_H
#define DEABERRATETRANSFORM_H
#include "basetransform.h"
#include <functional>
#include "lensfun/lensfun.h"
#include "./../Tools/mathtools.h"

struct CameraSettings;

class DeaberrateTransform : public BaseTransform
{
	std::shared_ptr<CameraSettings> _pCameraSettings;

	std::unique_ptr<lfDatabase> _pDatabase;	
	std::unique_ptr<lfModifier> _pModifier;

	template<PixelFormat pixelFormat>
	void CorrectDistortion()
	{
		using ChannelType = typename PixelFormatTraits<pixelFormat>::ChannelType;

		auto pSrcBitmap = std::static_pointer_cast<Bitmap<pixelFormat>>(_pSrcBitmap);
		auto pDstBitmap = std::static_pointer_cast<Bitmap<pixelFormat>>(_pDstBitmap);

		int lwidth = _pSrcBitmap->GetWidth() * 2 * 3;
		std::vector<float> pos(lwidth);

		for (uint32_t y = 0; y < _pSrcBitmap->GetHeight(); y++)
		{
			if (!_pModifier->ApplySubpixelGeometryDistortion(0.0, y, _pSrcBitmap->GetWidth(), 1, &pos[0]))
				throw std::runtime_error("unable to correct vignetting");

			auto pDstPixel = pDstBitmap->GetScanline(y);

			for (uint32_t x = 0; x < _pSrcBitmap->GetWidth(); ++x)
			{
				pDstPixel[0] = _pSrcBitmap->GetInterpolatedChannel(pos[6 * x], pos[6 * x + 1], 0);
				pDstPixel[1] = _pSrcBitmap->GetInterpolatedChannel(pos[6 * x + 2], pos[6 * x +  3], 1);
				pDstPixel[2] = _pSrcBitmap->GetInterpolatedChannel(pos[6 * x + 4], pos[6 * x + 5], 2);
				pDstPixel += 3;
			}
		}
	}

public:
	DeaberrateTransform(IBitmapPtr pSrcBitmap, std::shared_ptr<CameraSettings> pCameraSettings);
	void Run() override;
};

#endif
