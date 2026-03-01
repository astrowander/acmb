#include "MedianBlurTransform.h"

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <vector>

ACMB_NAMESPACE_BEGIN

// Histogram-based median helper for 8-bit channels
struct Histogram8
{
    std::array<int, 256> bins{};
    int count = 0;

    void reset()
    {
        bins.fill( 0 );
        count = 0;
    }

    void add( uint8_t val )
    {
        ++bins[val];
        ++count;
    }

    void remove( uint8_t val )
    {
        --bins[val];
        --count;
    }

    void addHistogram( const Histogram8& other )
    {
        for ( int i = 0; i < 256; ++i )
            bins[i] += other.bins[i];
        count += other.count;
    }

    void removeHistogram( const Histogram8& other )
    {
        for ( int i = 0; i < 256; ++i )
            bins[i] -= other.bins[i];
        count -= other.count;
    }

    uint8_t median() const
    {
        int target = ( count - 1 ) / 2;
        int cumulative = 0;
        for ( int i = 0; i < 256; ++i )
        {
            cumulative += bins[i];
            if ( cumulative > target )
                return static_cast<uint8_t>( i );
        }
        return 255;
    }
};

// Two-level histogram for 16-bit channels
struct Histogram16
{
    std::array<int, 256> coarse{};
    std::array<std::array<int, 256>, 256> fine{};
    int count = 0;

    void reset()
    {
        coarse.fill( 0 );
        for ( auto& f : fine )
            f.fill( 0 );
        count = 0;
    }

    void add( uint16_t val )
    {
        ++coarse[val >> 8];
        ++fine[val >> 8][val & 0xFF];
        ++count;
    }

    void remove( uint16_t val )
    {
        --coarse[val >> 8];
        --fine[val >> 8][val & 0xFF];
        --count;
    }

    void addHistogram( const Histogram16& other )
    {
        for ( int i = 0; i < 256; ++i )
        {
            if ( other.coarse[i] )
            {
                coarse[i] += other.coarse[i];
                for ( int j = 0; j < 256; ++j )
                    fine[i][j] += other.fine[i][j];
            }
        }
        count += other.count;
    }

    void removeHistogram( const Histogram16& other )
    {
        for ( int i = 0; i < 256; ++i )
        {
            if ( other.coarse[i] )
            {
                coarse[i] -= other.coarse[i];
                for ( int j = 0; j < 256; ++j )
                    fine[i][j] -= other.fine[i][j];
            }
        }
        count -= other.count;
    }

    uint16_t median() const
    {
        int target = ( count - 1 ) / 2;
        int cumulative = 0;
        for ( int i = 0; i < 256; ++i )
        {
            if ( cumulative + coarse[i] > target )
            {
                int innerCumulative = cumulative;
                for ( int j = 0; j < 256; ++j )
                {
                    innerCumulative += fine[i][j];
                    if ( innerCumulative > target )
                        return static_cast<uint16_t>( ( i << 8 ) | j );
                }
            }
            cumulative += coarse[i];
        }
        return 65535;
    }
};

template <typename ChannelType>
struct HistogramSelector;

template <>
struct HistogramSelector<uint8_t>
{
    using type = Histogram8;
};

template <>
struct HistogramSelector<uint16_t>
{
    using type = Histogram16;
};

template<PixelFormat pixelFormat>
class MedianBlurTransform_ : public MedianBlurTransform
{
    using ChannelType = typename PixelFormatTraits<pixelFormat>::ChannelType;
    static constexpr uint32_t channelCount = PixelFormatTraits<pixelFormat>::channelCount;
    static constexpr ChannelType channelMax = PixelFormatTraits<pixelFormat>::channelMax;

    using Hist = typename HistogramSelector<ChannelType>::type;

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

        auto pSrcBitmap = std::static_pointer_cast< Bitmap<pixelFormat> >( _pSrcBitmap );
        auto pDstBitmap = std::make_shared<Bitmap<pixelFormat>>( _pSrcBitmap->GetWidth(), _pSrcBitmap->GetHeight() );

        const int width = static_cast<int>( _pSrcBitmap->GetWidth() );
        const int height = static_cast<int>( _pSrcBitmap->GetHeight() );
        const int radius = ( _kernelSize - 1 ) / 2;

        // Process rows in parallel. Each thread uses column histograms + a running histogram.
        tbb::parallel_for( tbb::blocked_range<int>( 0, height ), [&] ( const tbb::blocked_range<int>& range )
        {
            // Per-column histograms for each channel
            std::vector<std::array<Hist, channelCount>> colHists( width );

            for ( int i = range.begin(); i < range.end(); ++i )
            {
                auto pDstScanline = pDstBitmap->GetScanline( i );

                const int top = std::max( 0, i - radius );
                const int bottom = std::min( height - 1, i + _kernelSize / 2 );

                if ( i == range.begin() )
                {
                    // Build column histograms from scratch for the first row of this range
                    for ( int x = 0; x < width; ++x )
                    {
                        for ( uint32_t ch = 0; ch < channelCount; ++ch )
                            colHists[x][ch].reset();

                        for ( int y = top; y <= bottom; ++y )
                        {
                            auto pPixel = pSrcBitmap->GetScanline( y ) + x * channelCount;
                            for ( uint32_t ch = 0; ch < channelCount; ++ch )
                                colHists[x][ch].add( pPixel[ch] );
                        }
                    }
                }
                else
                {
                    // Update column histograms incrementally: remove old top row, add new bottom row
                    const int prevTop = std::max( 0, ( i - 1 ) - radius );
                    const int prevBottom = std::min( height - 1, ( i - 1 ) + _kernelSize / 2 );

                    if ( top > prevTop )
                    {
                        // Remove row prevTop
                        for ( int x = 0; x < width; ++x )
                        {
                            auto pPixel = pSrcBitmap->GetScanline( prevTop ) + x * channelCount;
                            for ( uint32_t ch = 0; ch < channelCount; ++ch )
                                colHists[x][ch].remove( pPixel[ch] );
                        }
                    }

                    if ( bottom > prevBottom )
                    {
                        // Add row bottom
                        for ( int x = 0; x < width; ++x )
                        {
                            auto pPixel = pSrcBitmap->GetScanline( bottom ) + x * channelCount;
                            for ( uint32_t ch = 0; ch < channelCount; ++ch )
                                colHists[x][ch].add( pPixel[ch] );
                        }
                    }
                }

                // Now compute medians for each pixel in this row using sliding window over column histograms
                for ( uint32_t ch = 0; ch < channelCount; ++ch )
                {
                    Hist windowHist;
                    windowHist.reset();

                    const int initRight = std::min( width - 1, _kernelSize / 2 );

                    // Build initial window histogram for j=0
                    for ( int x = 0; x <= initRight; ++x )
                        windowHist.addHistogram( colHists[x][ch] );

                    pDstScanline[ch] = windowHist.median();

                    for ( int j = 1; j < width; ++j )
                    {
                        const int removeCol = j - radius - 1;
                        const int addCol = j + _kernelSize / 2;

                        if ( removeCol >= 0 )
                            windowHist.removeHistogram( colHists[removeCol][ch] );

                        if ( addCol < width )
                            windowHist.addHistogram( colHists[addCol][ch] );

                        pDstScanline[j * channelCount + ch] = windowHist.median();
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
