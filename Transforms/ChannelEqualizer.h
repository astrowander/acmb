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
	ChannelEqualizer(IBitmapPtr pSrcBitmap, const std::array< std::function<ChannelType( ChannelType )>, channelCount>& channelTransforms )
	: BaseChannelEqualizer(pSrcBitmap)
	, _channelTransforms(channelTransforms)
	{
	}

	void Run() override
	{
		_pDstBitmap = std::make_shared<Bitmap<pixelFormat>>(_pSrcBitmap->GetWidth(), _pSrcBitmap->GetHeight());
		auto pSrcBitmap = std::static_pointer_cast< Bitmap<pixelFormat> >( _pSrcBitmap );
		auto pDstBitmap = std::static_pointer_cast< Bitmap<pixelFormat> >( _pDstBitmap );

		oneapi::tbb::parallel_for( oneapi::tbb::blocked_range<int>( 0, _pSrcBitmap->GetHeight() ), [this, pSrcBitmap, pDstBitmap] ( const oneapi::tbb::blocked_range<int>& range )
		{
			for ( int i = range.begin(); i < range.end(); ++i )
			{
				for ( uint32_t ch = 0; ch < channelCount; ++ch )
				{
					auto pSrcScanline = pSrcBitmap->GetScanline( i ) + ch;
					auto pDstScanline = pDstBitmap->GetScanline( i ) + ch;

					for ( uint32_t x = 0; x < pSrcBitmap->GetWidth(); ++x )
					{
						pDstScanline[0] = _channelTransforms[ch]( pSrcScanline[0] );
						pSrcScanline += channelCount;
						pDstScanline += channelCount;
					}
				}
			}
		} );
	}
};


