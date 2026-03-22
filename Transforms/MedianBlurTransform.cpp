#include "MedianBlurTransform.h"

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

ACMB_NAMESPACE_BEGIN

class HistogramBase
{
public:
    virtual ~HistogramBase() = default;
    virtual void AddPixel(uint32_t value) = 0;
    virtual void RemovePixel(uint32_t value) = 0;
    virtual uint32_t GetMedian(size_t pixelCount) const = 0;
};

class Histogram8 : public HistogramBase
{
    std::vector<uint32_t> _histogram;

public:

    Histogram8()
    {
        _histogram.resize( 256 );
    }

    void AddPixel( uint32_t value ) override
    {
        ++_histogram[value];
    }

    void RemovePixel( uint32_t value ) override
    {
        --_histogram[value];
    }

    uint32_t GetMedian( size_t pixelCount ) const override
    {
        size_t count = 0;
        for ( size_t value = 0; value <= 255; ++value )
        {
            count += _histogram[value];
            if ( count > pixelCount / 2 )
            {
                return static_cast<uint32_t>( value );
            }
        }
        return 0;
    }
};

class Histogram16 : public HistogramBase
{
    std::vector<std::pair<uint32_t, std::vector<uint32_t>>> _histogram;
public:

    Histogram16()
    {
        _histogram.resize( 256 );
        for ( auto& pair : _histogram )
        {
            pair.first = 0;
            pair.second.resize( 256 );
        }
    }

    void AddPixel( uint32_t value ) override
    {
        uint32_t highByte = value >> 8;
        uint32_t lowByte = value & 0xFF;
        ++_histogram[highByte].second[lowByte];
        ++_histogram[highByte].first;
    }

    void RemovePixel( uint32_t value ) override
    {
        uint32_t highByte = value >> 8;
        uint32_t lowByte = value & 0xFF;
        --_histogram[highByte].second[lowByte];
        --_histogram[highByte].first;
    }

    uint32_t GetMedian( size_t pixelCount ) const override
    {
        size_t count = 0;
        for ( size_t highByte = 0; highByte <= 255; ++highByte )
        {
            count += _histogram[highByte].first;
            if ( count > pixelCount / 2 )
            {
                const std::vector<uint32_t>& lowBytesHistogram = _histogram[highByte].second;
                size_t lowCount = count - _histogram[highByte].first;
                for ( size_t lowByte = 0; lowByte <= 255; ++lowByte )
                {
                    lowCount += lowBytesHistogram[lowByte];
                    if ( lowCount > pixelCount / 2 )
                    {
                        return static_cast<uint32_t>( ( highByte << 8 ) | lowByte );
                    }
                }
            }
        }
        return 0;
    }
};


template<PixelFormat pixelFormat>
class MedianBlurTransform_ : public MedianBlurTransform
{
    using ChannelType = typename PixelFormatTraits<pixelFormat>::ChannelType;
    static constexpr uint32_t channelCount = PixelFormatTraits<pixelFormat>::channelCount;
    static constexpr ChannelType channelMax = PixelFormatTraits<pixelFormat>::channelMax;

    using HistogramType = std::conditional_t<sizeof(ChannelType) == 1, Histogram8, Histogram16>;

public:
    MedianBlurTransform_( std::shared_ptr<IBitmap> pSrcBitmap, int kernelSize )
        : MedianBlurTransform( pSrcBitmap, kernelSize )
    {}

    virtual void Run() override
    {
        if ( _kernelSize == 1 )
        {
            _pDstBitmap = _pSrcBitmap;
            return;
        }

        auto pSrcBitmap = std::static_pointer_cast< Bitmap<pixelFormat> >(_pSrcBitmap);
        auto pDstBitmap = std::make_shared<Bitmap<pixelFormat>>( _pSrcBitmap->GetWidth(), _pSrcBitmap->GetHeight() );

        tbb::parallel_for( tbb::blocked_range<int>( 0, _pSrcBitmap->GetHeight() ), [&] ( const tbb::blocked_range<int>& range )
        {
            for ( int i = range.begin(); i < range.end(); ++i )
            {
                ChannelType* pDstScanline = pDstBitmap->GetScanline( i );

                const int top = std::max( 0, i - ( _kernelSize - 1 ) / 2 );
                const int bottom = std::min<int>( _pSrcBitmap->GetHeight() - 1, i + _kernelSize / 2 );

                std::array<HistogramType, channelCount> histograms;

                int left = std::max(0,  - (_kernelSize - 1) / 2);
                int right = std::min<int>(_pSrcBitmap->GetWidth() - 1, _kernelSize / 2);

                /// initialize histograms for the first pixel in the row
                for ( int ch = 0; ch < channelCount; ++ch )
                {
                    HistogramType& histogram = histograms[ch];
                    for ( int y = top; y <= bottom; ++y )
                    {
                        auto pPixel = pSrcBitmap->GetScanline( y ) + left * channelCount + ch;
                        for ( int x = left; x <= right; ++x )
                        {
                            histogram.AddPixel( *pPixel );
                            pPixel += channelCount;
                        }
                    }
                }

                const size_t pixelCount = (right - left + 1) * (bottom - top + 1);

                /// find median values in histograms and set them to the first pixel in the row
                for ( int ch = 0; ch < channelCount; ++ch )
                {
                    *pDstScanline++ = histograms[ch].GetMedian(pixelCount);
                }

                for ( int j = 1; j < int( _pSrcBitmap->GetWidth() ); ++j )
                {
                    const int unboundedLeft = j - (_kernelSize - 1) / 2 - 1;
                    const int unboundedRight = j + _kernelSize / 2;
                    /// update histograms for the next pixel in the row
                    if ( unboundedLeft >= 0 )
                    {
                        for ( int y = top; y <= bottom; ++y )
                        {
                            auto pPixel = pSrcBitmap->GetScanline( y ) + unboundedLeft * channelCount;
                            for ( int ch = 0; ch < channelCount; ++ch )
                            {
                                histograms[ch].RemovePixel( pPixel[ch] );
                            }
                        }
                    }

                    if ( unboundedRight < int( _pSrcBitmap->GetWidth() ) )
                    {
                        for ( int y = top; y <= bottom; ++y )
                        {
                            auto pPixel = pSrcBitmap->GetScanline( y ) + unboundedRight * channelCount;
                            for ( int ch = 0; ch < channelCount; ++ch )
                            {
                                histograms[ch].AddPixel( pPixel[ch] );
                            }
                        }
                    }

                    left = std::max( 0, j - ( _kernelSize - 1 ) / 2 );
                    right = std::min<int>( _pSrcBitmap->GetWidth() - 1, j + _kernelSize / 2 );

                    const size_t pixelCount = (right - left + 1) * (bottom - top + 1);
                    /// find median values in histograms and set them to the first pixel in the row
                    for ( int ch = 0; ch < channelCount; ++ch )
                    {
                        *pDstScanline++ = histograms[ch].GetMedian(pixelCount);
                    }
                }
            }
        } );

        _pDstBitmap = pDstBitmap;
    }

    virtual void ValidateSettings() override
    {
        if ( _kernelSize < 1 )
            throw std::invalid_argument( "kernel size must be >= 1" );
    }
};

MedianBlurTransform::MedianBlurTransform( IBitmapPtr pSrcBitmap, int kernelSize )
: BaseTransform( pSrcBitmap )
, _kernelSize( kernelSize )
{}

std::shared_ptr<MedianBlurTransform> MedianBlurTransform::Create( IBitmapPtr pSrcBitmap, int kernelSize )
{
    if ( !pSrcBitmap )
        throw std::invalid_argument( "pSrcBitmap is null" );
    if ( kernelSize < 1 )
        throw std::invalid_argument( "kernel size must be >= 1" );

    switch ( pSrcBitmap->GetPixelFormat() )
    {
        case PixelFormat::Gray8:
            return std::make_shared<MedianBlurTransform_<PixelFormat::Gray8>>( pSrcBitmap, kernelSize );
        case PixelFormat::Gray16:
            return std::make_shared<MedianBlurTransform_<PixelFormat::Gray16>>( pSrcBitmap, kernelSize );
        case PixelFormat::RGB24:
            return std::make_shared<MedianBlurTransform_<PixelFormat::RGB24>>( pSrcBitmap, kernelSize );
        case PixelFormat::RGB48:
            return std::make_shared<MedianBlurTransform_<PixelFormat::RGB48>>( pSrcBitmap, kernelSize );
        default:
            throw std::invalid_argument( "unsupported pixel format" );
    }
}

std::shared_ptr<MedianBlurTransform> MedianBlurTransform::Create( PixelFormat pixelFormat, int kernelSize )
{
    switch ( pixelFormat )
    {
        case PixelFormat::Gray8:
            return std::make_shared<MedianBlurTransform_<PixelFormat::Gray8>>( nullptr, kernelSize );
        case PixelFormat::Gray16:
            return std::make_shared<MedianBlurTransform_<PixelFormat::Gray16>>( nullptr, kernelSize );
        case PixelFormat::RGB24:
            return std::make_shared<MedianBlurTransform_<PixelFormat::RGB24>>( nullptr, kernelSize );
        case PixelFormat::RGB48:
            return std::make_shared<MedianBlurTransform_<PixelFormat::RGB48>>( nullptr, kernelSize );
        default:
            throw std::invalid_argument( "unsupported pixel format" );
    }
}

IBitmapPtr MedianBlurTransform::MedianBlur( IBitmapPtr pSrcBitmap, int kernelSize )
{
    auto pTransform = MedianBlurTransform::Create( pSrcBitmap, kernelSize );
    return pTransform->RunAndGetBitmap();
}

ACMB_NAMESPACE_END
