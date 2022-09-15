#include "HistogramBuilder.h"
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

BaseHistorgamBuilder::BaseHistorgamBuilder(IBitmapPtr pBitmap, const Rect& roi)
: _pBitmap(pBitmap)
, _roi((roi.width&& roi.height) ? roi : Rect {0, 0, int(pBitmap->GetWidth()), int(pBitmap->GetHeight()) })
{
	
}

std::shared_ptr<BaseHistorgamBuilder> BaseHistorgamBuilder::Create(IBitmapPtr pBitmap, const Rect& roi)
{
	switch (pBitmap->GetPixelFormat())
	{
	case PixelFormat::Gray8:
		return std::make_shared<HistogramBuilder<PixelFormat::Gray8>>(pBitmap, roi);
	case PixelFormat::Gray16:
		return std::make_shared<HistogramBuilder<PixelFormat::Gray16>>(pBitmap, roi);
	case PixelFormat::RGB24:
		return std::make_shared<HistogramBuilder<PixelFormat::RGB24>>(pBitmap, roi);
	case PixelFormat::RGB48:
		return std::make_shared<HistogramBuilder<PixelFormat::RGB48>>(pBitmap, roi);
	default:
		throw std::runtime_error("pixel format should be known");
	}
}

template<PixelFormat pixelFormat>
HistogramBuilder<pixelFormat>::HistogramBuilder( IBitmapPtr pBitmap, const Rect& roi )
: BaseHistorgamBuilder( pBitmap, roi )
{
}

template<PixelFormat pixelFormat>
void HistogramBuilder<pixelFormat>::BuildHistogram()
{
	for ( uint32_t ch = 0; ch < channelCount; ++ch )
		_histograms[ch].resize( channelMax + 1 );

	oneapi::tbb::parallel_for( oneapi::tbb::blocked_range<int>( 0, _pBitmap->GetHeight() ), [this] ( const oneapi::tbb::blocked_range<int>& range )
	{
		for ( int i = range.begin(); i < range.end(); ++i )
		{
			std::scoped_lock lock( _mutex );
			for ( uint32_t ch = 0; ch < channelCount; ++ch )
			{
				auto pBitmap = std::static_pointer_cast< Bitmap<pixelFormat> >( _pBitmap );
				auto pChannel = pBitmap->GetScanline( _roi.y + i ) + _roi.x * channelCount + ch;

				for ( int x = 0; x < _roi.width; ++x )
				{
					ChannelType val = *pChannel;
					++_histograms[ch][val];
					if ( val < _statistics[ch].min )
					{
						_statistics[ch].min = val;
					}
					if ( val > _statistics[ch].max )
					{
						_statistics[ch].max = val;
					}
					if ( _histograms[ch][val] > _histograms[ch][_statistics[ch].peak] ||
						 ( _histograms[ch][val] == _histograms[ch][_statistics[ch].peak] && val > _statistics[ch].peak ) )
					{
						_statistics[ch].peak = val;
					}
					pChannel += channelCount;
				}
			}
		}
	} );

	const uint32_t pixCount = _roi.width * _roi.height;
	const uint32_t centilPixCount = pixCount / 100;
	for ( uint32_t ch = 0; ch < channelCount; ++ch )
	{
		uint32_t count = 0;
		uint32_t sum = 0;
		uint32_t curCentil = 1;
		for ( uint32_t i = 0; i < channelMax + 1; ++i )
		{
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

template<PixelFormat pixelFormat>
const BaseHistorgamBuilder::ChannelHistogram& HistogramBuilder<pixelFormat>::GetChannelHistogram( uint32_t ch ) const
{
	return _histograms[ch];
}

template<PixelFormat pixelFormat>
const HistogramStatistics& HistogramBuilder<pixelFormat>::GetChannelStatistics( uint32_t ch ) const
{
	return _statistics[ch];
}
