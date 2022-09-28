#include "ChannelEqualizer.h"
#include "HistogramBuilder.h"
#include "../Tools/mathtools.h"
#include <algorithm>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

ACMB_NAMESPACE_BEGIN

template<PixelFormat pixelFormat>
class ChannelEqualizer_ final : public ChannelEqualizer
{
	using ChannelType = typename PixelFormatTraits<pixelFormat>::ChannelType;
	static const uint32_t channelCount = PixelFormatTraits<pixelFormat>::channelCount;

	std::array<std::function<ChannelType( ChannelType )>, channelCount> _channelTransforms;

public:
	ChannelEqualizer_( IBitmapPtr pSrcBitmap, const std::array< std::function<ChannelType( ChannelType )>, channelCount>& channelTransforms )
		: ChannelEqualizer( pSrcBitmap )
		, _channelTransforms( channelTransforms )
	{
	}
	virtual void Run() override
	{
		_pDstBitmap = std::make_shared<Bitmap<pixelFormat>>( _pSrcBitmap->GetWidth(), _pSrcBitmap->GetHeight() );
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

ChannelEqualizer::ChannelEqualizer(IBitmapPtr pSrcBitmap)
: BaseTransform(pSrcBitmap)
{

}

std::shared_ptr<ChannelEqualizer> ChannelEqualizer::Create(IBitmapPtr pSrcBitmap, const std::vector< std::function<uint32_t( uint32_t )>>& channelTransforms )
{
	if ( !pSrcBitmap )
		throw std::invalid_argument( "pSrcBitmap is null" );

	if ( channelTransforms.size() != ChannelCount(pSrcBitmap->GetPixelFormat()))
		throw std::invalid_argument("Multiplier count must be equal to channel count");

	switch (pSrcBitmap->GetPixelFormat())
	{
	case PixelFormat::Gray8:
		return std::make_shared<ChannelEqualizer_<PixelFormat::Gray8>>( pSrcBitmap, std::array< std::function<uint8_t( uint8_t )>, 1>
		{ 
			[channelTransforms]( uint8_t arg )
			{
				return uint8_t( std::clamp( channelTransforms[0]( arg ), 0u, 255u ) );
			}
		} );
	case PixelFormat::Gray16:
		return std::make_shared<ChannelEqualizer_<PixelFormat::Gray16>>( pSrcBitmap, std::array< std::function<uint16_t( uint16_t )>, 1>
		{
			[channelTransforms]( uint16_t arg )
			{
				return uint16_t( std::clamp( channelTransforms[0]( arg ), 0u, 65535u ) );
			}
		} );
	case PixelFormat::RGB24:
		return std::make_shared<ChannelEqualizer_<PixelFormat::RGB24>>( pSrcBitmap, std::array< std::function<uint8_t( uint8_t )>, 3>
		{
			[channelTransforms]( uint8_t arg )
			{
				return uint8_t( std::clamp( channelTransforms[0]( arg ), 0u, 255u ) );
			},
				[channelTransforms] ( uint8_t arg )
			{
				return uint8_t( std::clamp( channelTransforms[1]( arg ), 0u, 255u ) );
			},
				[channelTransforms] ( uint8_t arg )
			{
				return uint8_t( std::clamp( channelTransforms[2]( arg ), 0u, 255u ) );
			}

		} );
	case PixelFormat::RGB48:
		return std::make_shared<ChannelEqualizer_<PixelFormat::RGB48>>( pSrcBitmap, std::array< std::function<uint16_t( uint16_t )>, 3>
		{
			[channelTransforms]( uint16_t arg )
			{
				return uint16_t( std::clamp( channelTransforms[0]( arg ), 0u, 65535u ) );
			},
				[channelTransforms] ( uint16_t arg )
			{
				return uint16_t( std::clamp( channelTransforms[1]( arg ), 0u, 65535u ) );
			},
				[channelTransforms] ( uint16_t arg )
			{
				return uint16_t( std::clamp( channelTransforms[2]( arg ), 0u, 65535u ) );
			}

		} );
	default:
		throw std::runtime_error("pixel format should be known");
	}
}

IBitmapPtr ChannelEqualizer::AutoEqualize( IBitmapPtr pSrcBitmap )
{
	auto pHistBuilder = HistorgamBuilder::Create( pSrcBitmap );
	pHistBuilder->BuildHistogram();
	std::unique_ptr<ChannelEqualizer> pEqualizer;
	switch ( pSrcBitmap->GetPixelFormat() )
	{
		case PixelFormat::Gray8:		
			pEqualizer = std::make_unique<ChannelEqualizer_<PixelFormat::Gray8>>( pSrcBitmap, std::array< std::function<uint8_t( uint8_t )>, 1>
			{
				[pHistBuilder]( uint8_t arg )
				{
					const float min = pHistBuilder->GetChannelStatistics( 0 ).min;
					const float med = pHistBuilder->GetChannelStatistics( 0 ).centils[50];
					const float max = pHistBuilder->GetChannelStatistics( 0 ).max;

					return std::clamp(ArbitraryQuadraticInterpolation( arg, min, 0, med, 60, max, 255 ), 0.0f, 255.0f);
				}
			} );
			break;
		case PixelFormat::Gray16:
			pEqualizer = std::make_unique<ChannelEqualizer_<PixelFormat::Gray16>>( pSrcBitmap, std::array< std::function<uint16_t( uint16_t )>, 1>
			{
				[pHistBuilder]( uint16_t arg )
				{
					const float min = pHistBuilder->GetChannelStatistics( 0 ).min;
					const float med = pHistBuilder->GetChannelStatistics( 0 ).centils[50];
					const float max = pHistBuilder->GetChannelStatistics( 0 ).max;

					return std::clamp( ArbitraryQuadraticInterpolation( arg, min, 0, med, 15360, max, 65535 ), 0.0f, 65535.0f );
				}
			} );
			break;
		case PixelFormat::RGB24:
			pEqualizer = std::make_unique<ChannelEqualizer_<PixelFormat::RGB24>>( pSrcBitmap, std::array< std::function<uint8_t( uint8_t )>, 3>
			{
				[pHistBuilder]( uint8_t arg )
				{
					const float min = pHistBuilder->GetChannelStatistics( 0 ).min;
					const float med = pHistBuilder->GetChannelStatistics( 0 ).centils[50];
					const float max = pHistBuilder->GetChannelStatistics( 0 ).max;

					return std::clamp( ArbitraryQuadraticInterpolation( arg, min, 0, med, 60, max, 255 ), 0.0f, 255.0f );
				},
				[pHistBuilder] ( uint8_t arg )
				{
					const float min = pHistBuilder->GetChannelStatistics( 1 ).min;
					const float med = pHistBuilder->GetChannelStatistics( 1 ).centils[50];
					const float max = pHistBuilder->GetChannelStatistics( 1 ).max;

					return std::clamp( ArbitraryQuadraticInterpolation( arg, min, 0, med, 60, max, 255 ), 0.0f, 255.0f );
				},
				[pHistBuilder] ( uint8_t arg )
				{
					const float min = pHistBuilder->GetChannelStatistics( 2 ).min;
					const float med = pHistBuilder->GetChannelStatistics( 2 ).centils[50];
					const float max = pHistBuilder->GetChannelStatistics( 2 ).max;

					return std::clamp( ArbitraryQuadraticInterpolation( arg, min, 0, med, 60, max, 255 ), 0.0f, 255.0f );
				}
			} );
			break;
		case PixelFormat::RGB48:
			pEqualizer = std::make_unique<ChannelEqualizer_<PixelFormat::RGB48>>( pSrcBitmap, std::array< std::function<uint16_t( uint16_t )>, 3>
			{
				[pHistBuilder]( uint16_t arg )
				{
					return std::clamp( ArbitraryQuadraticInterpolation( arg, pHistBuilder->GetChannelStatistics( 0 ).min, 0, pHistBuilder->GetChannelStatistics( 0 ).centils[50], 15360, pHistBuilder->GetChannelStatistics( 0 ).max, 65535 ), 0.0f, 65535.0f );
				},
					[pHistBuilder] ( uint16_t arg )
				{
					return std::clamp( ArbitraryQuadraticInterpolation( arg, pHistBuilder->GetChannelStatistics( 1 ).min, 0, pHistBuilder->GetChannelStatistics( 1 ).centils[50], 15360, pHistBuilder->GetChannelStatistics( 1 ).max, 65535 ), 0.0f, 65535.0f );
				},
					[pHistBuilder] ( uint16_t arg )
				{
					return std::clamp( ArbitraryQuadraticInterpolation( arg, pHistBuilder->GetChannelStatistics( 2 ).min, 0, pHistBuilder->GetChannelStatistics( 2 ).centils[50], 15360, pHistBuilder->GetChannelStatistics( 2 ).max, 65535 ), 0.0f, 65535.0f );
				}
			} );
			break;
		default:
			throw std::runtime_error( "Pixel format sholud be known" );
		
	}
	
	return pEqualizer->RunAndGetBitmap();
}

AutoChannelEqualizer::AutoChannelEqualizer( IBitmapPtr pSrcBitmap )
: BaseTransform( pSrcBitmap )
{

}

void AutoChannelEqualizer::Run()
{
	_pDstBitmap = ChannelEqualizer::AutoEqualize( _pSrcBitmap );
}

ACMB_NAMESPACE_END
