#include "BitmapDivisor.h"
#include "HistogramBuilder.h"
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <parallel_hashmap/phmap.h>

#include <algorithm>
#include <array>

using phmap::parallel_flat_hash_map;

ACMB_NAMESPACE_BEGIN

template <PixelFormat pixelFormat>
class BitmapDivisor_ final : public BitmapDivisor
{
    using ChannelType = typename PixelFormatTraits<pixelFormat>::ChannelType;    

    inline static parallel_flat_hash_map<IBitmapPtr, std::array<float, 4>> _cachedPivots = {};

public:

    BitmapDivisor_( IBitmapPtr pSrcBitmap, const Settings& settings )
    : BitmapDivisor( pSrcBitmap, settings )
    {
    }

    virtual void Run() override
    {
        auto pSrcBitmap = std::static_pointer_cast< Bitmap<pixelFormat> >( _pSrcBitmap );
        auto pDivisor = std::static_pointer_cast< Bitmap<pixelFormat> >( _settings.pDivisor );
        constexpr uint32_t channelCount = PixelFormatTraits<pixelFormat>::channelCount;
        using ChannelType = typename PixelFormatTraits<pixelFormat>::ChannelType;

        const float srcBlackLevel = pSrcBitmap->GetCameraSettings() ? pSrcBitmap->GetCameraSettings()->blackLevel : 0;
        std::array<float, 4> pivots = {};

        if ( _cachedPivots.contains( pDivisor ) )
        {
            pivots = _cachedPivots[pDivisor];
        }
        else
        {
            auto pHistogramBuilder = HistorgamBuilder::Create( pDivisor );
            pHistogramBuilder->BuildHistogram();
            pivots[0] = srcBlackLevel + ( pHistogramBuilder->GetChannelStatistics( 0 ).centils[99] - srcBlackLevel ) / ( pSrcBitmap->GetCameraSettings() ? pSrcBitmap->GetCameraSettings()->channelPremultipiers[0] : 1.0f );
            pivots[1] = srcBlackLevel + ( pHistogramBuilder->GetChannelStatistics( 0 ).centils[99] - srcBlackLevel ) / ( pSrcBitmap->GetCameraSettings() ? pSrcBitmap->GetCameraSettings()->channelPremultipiers[1] : 1.0f );
            pivots[2] = srcBlackLevel + ( pHistogramBuilder->GetChannelStatistics( 0 ).centils[99] - srcBlackLevel ) / ( pSrcBitmap->GetCameraSettings() ? pSrcBitmap->GetCameraSettings()->channelPremultipiers[1] : 1.0f );
            pivots[3] = srcBlackLevel + ( pHistogramBuilder->GetChannelStatistics( 0 ).centils[99] - srcBlackLevel ) / ( pSrcBitmap->GetCameraSettings() ? pSrcBitmap->GetCameraSettings()->channelPremultipiers[2] : 1.0f );
            _cachedPivots[pDivisor] = pivots;
        }

        const float maxChannel = pSrcBitmap->GetCameraSettings() ? float( pSrcBitmap->GetCameraSettings()->maxChannel ) : float( PixelFormatTraits<pixelFormat>::channelMax );

        oneapi::tbb::parallel_for( oneapi::tbb::blocked_range<int>( 0, _pSrcBitmap->GetHeight() ), [&] ( const oneapi::tbb::blocked_range<int>& range )
        {
            for ( int y = range.begin(); y < range.end(); ++y )
            {
                auto pSrcScanline = pSrcBitmap->GetScanline( y );
                auto pDivisorScanline = pDivisor->GetScanline( y );

                for ( uint32_t x = 0; x < pSrcBitmap->GetWidth(); ++x )
                {
                    const auto subpixelIndex = ( x % 2 ) + ( 2 * ( y % 2 ) );                        
                    const size_t index = x * channelCount;
                    const float srcValue = pSrcScanline[index];
                    const float divisorValue = pDivisorScanline[index];
                    float coeff = std::max( 1.0f, ( pivots[subpixelIndex] - srcBlackLevel ) / ( divisorValue - srcBlackLevel ) );
                    coeff = 1.0f + ( coeff - 1.0f ) * ( _settings.intensity / 100.0f );
                    const float res = std::min( srcBlackLevel + std::max( 0.0f, srcValue - srcBlackLevel ) * coeff, maxChannel );
                    pSrcScanline[index] = ChannelType( res );
                }
            }
        } );
        this->_pDstBitmap = this->_pSrcBitmap;
    }

    virtual void ValidateSettings() override
    {
        if ( _pSrcBitmap->GetPixelFormat() != _settings.pDivisor->GetPixelFormat() )
            throw std::invalid_argument( "bitmaps should have the same pixel format" );

        if ( _pSrcBitmap->GetWidth() != _settings.pDivisor->GetWidth() || _pSrcBitmap->GetHeight() != _settings.pDivisor->GetHeight() )
            throw std::invalid_argument( "bitmaps should have the same size" );
    }
};

BitmapDivisor::BitmapDivisor( IBitmapPtr pSrcBitmap, const Settings& settings )
    : BaseTransform( pSrcBitmap )
    , _settings( settings )
{
}

std::shared_ptr<BitmapDivisor> BitmapDivisor::Create( IBitmapPtr pSrcBitmap, const Settings& settings )
{
    if ( !pSrcBitmap )
        throw std::invalid_argument( "pSrcBitmap is null" );

    if ( !settings.pDivisor )
        throw std::invalid_argument( "pDivisor is null" );

    if ( pSrcBitmap->GetPixelFormat() != settings.pDivisor->GetPixelFormat() )
        throw std::invalid_argument( "bitmaps should have the same pixel format" );

    if ( pSrcBitmap->GetWidth() != settings.pDivisor->GetWidth() || pSrcBitmap->GetHeight() != settings.pDivisor->GetHeight() )
        throw std::invalid_argument( "bitmaps should have the same size" );

    switch ( pSrcBitmap->GetPixelFormat() )
    {
        case PixelFormat::Gray8:
            return std::make_shared<BitmapDivisor_<PixelFormat::Gray8>>( pSrcBitmap, settings );
        case PixelFormat::Gray16:
            return std::make_shared<BitmapDivisor_<PixelFormat::Gray16>>( pSrcBitmap, settings );
        case PixelFormat::Bayer16:
            return std::make_shared<BitmapDivisor_<PixelFormat::Gray16>>( pSrcBitmap, settings );
        default:
            throw std::runtime_error( "only grayscale bitmaps can be divided" );
    }
}

std::shared_ptr<BitmapDivisor> BitmapDivisor::Create( PixelFormat srcPixelFormat, const Settings& settings )
{
    if ( !settings.pDivisor )
        throw std::invalid_argument( "pDivisor is null" );

    switch ( srcPixelFormat )
    {
        case PixelFormat::Gray8:
            return std::make_shared<BitmapDivisor_<PixelFormat::Gray8>>( nullptr, settings );
        case PixelFormat::Gray16:
            return std::make_shared<BitmapDivisor_<PixelFormat::Gray16>>( nullptr, settings );
        case PixelFormat::Bayer16:
            return std::make_shared<BitmapDivisor_<PixelFormat::Gray16>>( nullptr, settings );
        default:
            throw std::runtime_error( "only grayscale bitmaps can be divided" );
    }
}

IBitmapPtr BitmapDivisor::Divide( IBitmapPtr pSrcBitmap, const Settings& settings )
{
    auto pSubtractor = Create( pSrcBitmap, settings );
    return pSubtractor->RunAndGetBitmap();
}

ACMB_NAMESPACE_END