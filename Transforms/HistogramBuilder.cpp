#include "HistogramBuilder.h"
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/enumerable_thread_specific.h>
#include <array>
#include <mutex>

#undef min
#undef max

ACMB_NAMESPACE_BEGIN

template <PixelFormat pixelFormat>
class HistogramBuilder_ final : public HistorgamBuilder
{
	using ChannelType = typename PixelFormatTraits<pixelFormat>::ChannelType;
	static const auto channelCount = PixelFormatTraits<pixelFormat>::channelCount;
	static const uint32_t channelMax = PixelFormatTraits<pixelFormat>::channelMax;

	std::array<ChannelHistogram, channelCount> _histograms;
	std::array<HistogramStatistics, channelCount> _statistics;

public:
	HistogramBuilder_( IBitmapPtr pBitmap, const Rect& roi )
	: HistorgamBuilder( pBitmap, roi )
	{
	}

	virtual void BuildHistogram() override
	{
		for ( uint32_t ch = 0; ch < channelCount; ++ch )
			_histograms[ch].resize( channelMax + 1 );
		
		struct ThreadLocalData
		{
			std::array<ChannelHistogram, channelCount > histograms;
            std::array<HistogramStatistics, channelCount > statistics;

			ThreadLocalData()
			{
                for ( uint32_t ch = 0; ch < channelCount; ++ch )
                {
                    histograms[ch].resize( channelMax + 1 );
                }
			}
		};

        tbb::enumerable_thread_specific<ThreadLocalData> tld;

		oneapi::tbb::parallel_for( oneapi::tbb::blocked_range<int>( 0, _pBitmap->GetHeight() ), [&] ( const oneapi::tbb::blocked_range<int>& range )
		{
			auto& local = tld.local();
			for ( int i = range.begin(); i < range.end(); ++i )
			{
				for ( uint32_t ch = 0; ch < channelCount; ++ch )
				{
					auto pBitmap = std::static_pointer_cast< Bitmap<pixelFormat> >( _pBitmap );
					auto pChannel = pBitmap->GetScanline( _roi.y + i ) + _roi.x * channelCount + ch;

					for ( int x = 0; x < _roi.width; ++x )
					{
						ChannelType val = *pChannel;
						++local.histograms[ch][val];
                        local.statistics[ch].min = std::min<ChannelType>( local.statistics[ch].min, val );
                        local.statistics[ch].max = std::max<ChannelType>( local.statistics[ch].max, val );
						pChannel += channelCount;
					}
				}
			}
		} );

		for ( auto& local : tld )
		{
			for ( uint32_t ch = 0; ch < channelCount; ++ch )
			{
                for ( uint32_t val = 0; val <= channelMax; ++val )
                {
                    _histograms[ch][val] += local.histograms[ch][val];
                    _statistics[ch].min = std::min( _statistics[ch].min, local.statistics[ch].min );
                    _statistics[ch].max = std::max( _statistics[ch].max, local.statistics[ch].max );
                }
			}
		}

		const uint32_t pixCount = _roi.width * _roi.height;
		const uint32_t centilPixCount = pixCount / 100;
		for ( uint32_t ch = 0; ch < channelCount; ++ch )
		{
			uint32_t count = 0;
			uint32_t sum = 0;
			uint32_t curCentil = 1;
			for ( uint32_t i = 0; i < channelMax + 1; ++i )
			{
                if ( _histograms[ch][i] >= _histograms[ch][_statistics[ch].peak])
					_statistics[ch].peak = i;

				count += _histograms[ch][i];
				sum += i * _histograms[ch][i];
				while ( count > centilPixCount )
				{
					_statistics[ch].centils[curCentil] = i;
					++curCentil;
					count -= centilPixCount;
				}
			}

			_statistics[ch].mean = float( sum ) / float( pixCount );
			for ( uint32_t i = 0; i < channelMax + 1; ++i )
			{
				_statistics[ch].dev += _histograms[ch][i] * ( i - _statistics[ch].mean ) * ( i - _statistics[ch].mean );
			}

			_statistics[ch].dev /= float( pixCount );
			_statistics[ch].dev = sqrt( _statistics[ch].dev );
		}
	}

	virtual const ChannelHistogram& GetChannelHistogram( uint32_t ch ) const override
	{
		return _histograms[ch];
	}

	virtual const HistogramStatistics& GetChannelStatistics( uint32_t ch ) const override
	{
		return _statistics[ch];
	}
};

HistorgamBuilder::HistorgamBuilder(IBitmapPtr pBitmap, const Rect& roi)
: _pBitmap(pBitmap)
, _roi((roi.width&& roi.height) ? roi : Rect {0, 0, int(pBitmap->GetWidth()), int(pBitmap->GetHeight()) })
{
	
}

std::shared_ptr<HistorgamBuilder> HistorgamBuilder::Create(IBitmapPtr pBitmap, const Rect& roi)
{
	if ( !pBitmap )
		throw std::invalid_argument( "pSrcBitmap is null" );

	switch (pBitmap->GetPixelFormat())
	{
	case PixelFormat::Gray8:
		return std::make_shared<HistogramBuilder_<PixelFormat::Gray8>>(pBitmap, roi);
	case PixelFormat::Gray16:
		return std::make_shared<HistogramBuilder_<PixelFormat::Gray16>>(pBitmap, roi);
	case PixelFormat::Bayer16:
		return std::make_shared<HistogramBuilder_<PixelFormat::Bayer16>>( pBitmap, roi );
	case PixelFormat::RGB24:
		return std::make_shared<HistogramBuilder_<PixelFormat::RGB24>>(pBitmap, roi);
	case PixelFormat::RGB48:
		return std::make_shared<HistogramBuilder_<PixelFormat::RGB48>>(pBitmap, roi);
	default:
		throw std::runtime_error("pixel format should be known");
	}
}

std::shared_ptr<HistorgamBuilder> HistorgamBuilder::Create( PixelFormat pixelFormat, const Rect& roi )
{
	switch ( pixelFormat )
	{
		case PixelFormat::Gray8:
			return std::make_shared<HistogramBuilder_<PixelFormat::Gray8>>( nullptr, roi );
		case PixelFormat::Gray16:
			return std::make_shared<HistogramBuilder_<PixelFormat::Gray16>>( nullptr, roi );
		case PixelFormat::Bayer16:
			return std::make_shared<HistogramBuilder_<PixelFormat::Bayer16>>( nullptr, roi );
		case PixelFormat::RGB24:
			return std::make_shared<HistogramBuilder_<PixelFormat::RGB24>>( nullptr, roi );
		case PixelFormat::RGB48:
			return std::make_shared<HistogramBuilder_<PixelFormat::RGB48>>( nullptr, roi );
		default:
			throw std::runtime_error( "pixel format should be known" );
	}
}

ACMB_NAMESPACE_END
