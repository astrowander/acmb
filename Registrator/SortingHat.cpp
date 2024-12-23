#include "SortingHat.h"

#include "./../Transforms/LaplacianTransform.h"
#include "./../Transforms/HistogramBuilder.h"
#include "./../Transforms/converter.h"

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/enumerable_thread_specific.h>
#include <numbers>

#undef min
#undef max

ACMB_NAMESPACE_BEGIN

template<PixelFormat pixelFormat>
float GetBgLuminance( std::shared_ptr<Bitmap<pixelFormat>> pBitmap )
{
    using ChannelType = typename PixelFormatTraits<pixelFormat>::ChannelType;
    constexpr float cBgSquareFraction = 0.1464466f;

    const int width = pBitmap->GetWidth();
    const int bgSquareSize = int( width * cBgSquareFraction + 0.5f );

    oneapi::tbb::enumerable_thread_specific<float> totalLuminancePerThread;
    oneapi::tbb::parallel_for( oneapi::tbb::blocked_range<uint32_t>( 0u, bgSquareSize ), [&] ( const oneapi::tbb::blocked_range<uint32_t>& range )
    {
        auto& tlLocal = totalLuminancePerThread.local();

        for ( uint32_t i = range.begin(); i < range.end(); ++i )
        {
            ChannelType* pTopScanline = ( ChannelType* ) pBitmap->GetPlanarScanline( i );
            ChannelType* pBottomScanline = ( ChannelType* ) pBitmap->GetPlanarScanline( width - i - 1 );

            for ( int j = 0; j < bgSquareSize; ++j )
            {
                tlLocal += pTopScanline[j];
                tlLocal += pTopScanline[width - j - 1];
                tlLocal += pBottomScanline[j];
                tlLocal += pBottomScanline[width - j - 1];
            }
        }
    } );

    float bgLuminance = 0.0f;
    for ( auto& tls : totalLuminancePerThread )
        bgLuminance += tls;

    bgLuminance /= 4 * bgSquareSize * bgSquareSize;

    return bgLuminance;
}

template<PixelFormat pixelFormat>
float GetTotalLuminance( std::shared_ptr<Bitmap<pixelFormat>> pBitmap, float bgLuminance )
{
    using ChannelType = typename PixelFormatTraits<pixelFormat>::ChannelType;

    oneapi::tbb::enumerable_thread_specific<float> totalLuminancePerThread;
    oneapi::tbb::parallel_for( oneapi::tbb::blocked_range<uint32_t>( 0u, pBitmap->GetHeight() ), [&] ( const oneapi::tbb::blocked_range<uint32_t>& range )
    {
        auto& tlLocal = totalLuminancePerThread.local();

        for ( uint32_t i = range.begin(); i < range.end(); ++i )
        {
            ChannelType* pScanline = ( ChannelType* ) pBitmap->GetPlanarScanline( i );

            for ( uint32_t j = 0; j < pBitmap->GetWidth(); ++j )
            {
                tlLocal += std::max( 0.0f, pScanline[j] - bgLuminance );
            }
        }
    } );

    float totalLuminance = 0.0f;
    for ( auto& tls : totalLuminancePerThread )
        totalLuminance += tls;

    return totalLuminance;
}

template<PixelFormat pixelFormat>
float GetLuminanceInsideRadius( std::shared_ptr<Bitmap<pixelFormat>> pBitmap, float radius, float bgLuminance )
{
    using ChannelType = typename PixelFormatTraits<pixelFormat>::ChannelType;
    const float halfSize = pBitmap->GetWidth() * 0.5f;

    oneapi::tbb::enumerable_thread_specific<float> totalLuminancePerThread;
    oneapi::tbb::parallel_for( oneapi::tbb::blocked_range<uint32_t>( 0u, pBitmap->GetHeight() ), [&] ( const oneapi::tbb::blocked_range<uint32_t>& range )
    {
        auto& tlLocal = totalLuminancePerThread.local();
        for ( uint32_t i = range.begin(); i < range.end(); ++i )
        {
            const float yPos = i - halfSize + 0.5f;
            const float closeY = yPos > 0 ? yPos - 0.5f : yPos + 0.5f;
            const float farY = yPos > 0 ? yPos + 0.5f : yPos - 0.5f;

            ChannelType* pScanline = ( ChannelType* ) pBitmap->GetPlanarScanline( i );

            for ( uint32_t j = 0; j < pBitmap->GetWidth(); ++j )
            {
                const float xPos = j - halfSize + 0.5f;
                const float closeX = xPos > 0 ? xPos - 0.5f : xPos + 0.5f;
                const float farX = xPos > 0 ? xPos + 0.5f : xPos - 0.5f;

                const float closeDist = std::sqrt( closeX * closeX + closeY * closeY );
                const float farDist = std::sqrt( farX * farX + farY * farY );

                if ( farDist < radius )
                {
                    tlLocal += pScanline[j] - bgLuminance;
                }
                else if ( closeDist < radius )
                {
                    tlLocal += ( pScanline[j] - bgLuminance ) * 0.5f;
                }
            }
        }
    } );

    float totalLuminance = 0.0f;
    for ( auto& tls : totalLuminancePerThread )
        totalLuminance += tls;

    return totalLuminance;
}
template<PixelFormat pixelFormat>
float GetContentRadius( std::shared_ptr<Bitmap<pixelFormat>> pBitmap )
{
    using ChannelType = typename PixelFormatTraits<pixelFormat>::ChannelType;
    const float halfSize = pBitmap->GetWidth() * 0.5f;

    float minRadius = 0;
    float maxRadius = halfSize;
    
    const float bgLuminance = GetBgLuminance( pBitmap );
    const float totalLuminance = GetLuminanceInsideRadius( pBitmap, maxRadius, bgLuminance );
    
    float minLuminance = 0;
    float maxLuminance = totalLuminance;

    while ( maxRadius - minRadius > 1.0f )
    {
        const float radius = ( minRadius + maxRadius ) * 0.5f;
        const float estimatedOuterLuminance = 2.0f * std::numbers::pi * (maxRadius * maxRadius - radius * radius) * bgLuminance;

        const float luminance = GetLuminanceInsideRadius( pBitmap, radius, bgLuminance );
        const float outerLuminance = maxLuminance - luminance;
        if ( estimatedOuterLuminance < outerLuminance )
        {
            minRadius = radius;
            minLuminance = luminance;
        }
        else
        {
            maxRadius = radius;
            maxLuminance = luminance;
        }
    }

    return (minRadius + maxRadius) * 0.5f;
}

SortingHat::SortingHat( const ImageParams& imageParams )
: _imageParams( imageParams)
{
}

void SortingHat::AddFrame( IBitmapPtr pBitmap )
{
    if ( !pBitmap )
        throw std::runtime_error( "SortingHat: invalid bitmap" );

    const auto grayFormat = ConstructPixelFormat( BitsPerChannel( _imageParams.GetPixelFormat() ), 1 );    

    auto pLaplacian = ConvolutionTransform::ApplyLaplacian( Converter::Convert( pBitmap, grayFormat ) );
    const auto width = pLaplacian->GetWidth();
    const auto height = pLaplacian->GetHeight();

    float key = 0;    

    auto pHistogramBuilder = HistogramBuilder::Create( pLaplacian );
    pHistogramBuilder->BuildHistogram( Rect( 0, 0, width, height / 2 ) );
    key = pHistogramBuilder->GetChannelStatistics(0).mean + pHistogramBuilder->GetChannelStatistics(0).dev;

    pHistogramBuilder->BuildHistogram( Rect( 0, 0, width, height / 2 ) );
    key = std::min( pHistogramBuilder->GetChannelStatistics(0).mean + pHistogramBuilder->GetChannelStatistics(0).dev, key );

    pHistogramBuilder->BuildHistogram( Rect( 0, height / 2, width / 2, height - height / 2 ) );
    key = std::min( pHistogramBuilder->GetChannelStatistics(0).mean + pHistogramBuilder->GetChannelStatistics(0).dev, key );

    pHistogramBuilder->BuildHistogram( Rect( width / 2, height / 2, width - width / 2, height - height / 2 ) );
    key = std::min( pHistogramBuilder->GetChannelStatistics(0).mean + pHistogramBuilder->GetChannelStatistics(0).dev, key );

    _frames.insert_or_assign( key, Frame{ .pBitmap = pLaplacian, .tempFilePath = {}, .index = uint32_t( _frames.size() ) } );
}

std::vector<SortingHat::Frame> SortingHat::GetBestFrames( uint32_t frameCount ) const
{
    if ( frameCount == 0 )
        return std::vector<Frame>{};

    if ( frameCount > _frames.size() )
        throw std::out_of_range( "frameCount exceeds the number of frames" );

    std::vector<Frame> bestFrames;
    bestFrames.reserve( frameCount );

    for ( const auto& frame : _frames )
    {
        bestFrames.push_back( frame.second );
        if ( bestFrames.size() >= frameCount )
            break;
    }
    return bestFrames;
}

std::vector<SortingHat::Frame> SortingHat::GetBestFramesByPercentage( float percentage ) const
{
    const auto frameCount = std::ceil( _frames.size() * percentage );
    return GetBestFrames( frameCount );
}

std::vector<SortingHat::Frame> SortingHat::GetBestFramesByQualityThreshold( float qualityThreshold ) const
{
    if ( _frames.empty() )
        return {};

    const float bestQuality = _frames.begin()->first;
    const float worstQuality = _frames.rbegin()->first;

    uint32_t frameCount = 0;
    for ( const auto& frame : _frames )
    {
        if ( frame.first < worstQuality + ( bestQuality - worstQuality ) * qualityThreshold )
            break;

        ++frameCount;
    }

    return GetBestFrames( frameCount );
}

uint32_t SortingHat::GetFrameCount() const
{
    return uint32_t( _frames.size() );
}

ACMB_NAMESPACE_END
