#include "BitmapDivisor.h"
#include "HistogramBuilder.h"
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <algorithm>

ACMB_NAMESPACE_BEGIN

template <PixelFormat pixelFormat>
class BitmapDivisor_ final : public BitmapDivisor
{
    using ChannelType = typename PixelFormatTraits<pixelFormat>::ChannelType;
    std::shared_ptr<HistorgamBuilder> _pHistogramBuilder;
public:

    BitmapDivisor_( IBitmapPtr pSrcBitmap, IBitmapPtr pDivisor )
    : BitmapDivisor( pSrcBitmap, pDivisor )
    {
        _pHistogramBuilder = HistorgamBuilder::Create( _pSrcBitmap );
    }

    virtual void Run() override
    {
        auto pSrcBitmap = std::static_pointer_cast< Bitmap<pixelFormat> >( _pSrcBitmap );
        auto pDivisor = std::static_pointer_cast< Bitmap<pixelFormat> >( _pDivisor );
        constexpr uint32_t channelCount = PixelFormatTraits<pixelFormat>::channelCount;
        using ChannelType = typename PixelFormatTraits<pixelFormat>::ChannelType;

        _pHistogramBuilder->BuildHistogram();
        int maxValues[channelCount] = {};
        for ( uint32_t ch = 0; ch < channelCount; ++ch )
        {
            maxValues[ch] = _pHistogramBuilder->GetChannelStatistics(ch).centils[99];
        }

        const auto srcBlackLevel = pSrcBitmap->GetCameraSettings() ? pSrcBitmap->GetCameraSettings()->blackLevel : 0;
        const auto divisorBlackLevel = pDivisor->GetCameraSettings() ? pDivisor->GetCameraSettings()->blackLevel : 0;
        const auto diffLevels = srcBlackLevel - divisorBlackLevel;
        oneapi::tbb::parallel_for( oneapi::tbb::blocked_range<int>( 0, _pSrcBitmap->GetHeight() ), [pSrcBitmap, pDivisor, maxValues, diffLevels] ( const oneapi::tbb::blocked_range<int>& range )
        {
            for ( int i = range.begin(); i < range.end(); ++i )
            {
                auto pSrcScanline = pSrcBitmap->GetScanline( i );
                auto pDivisorScanline = pDivisor->GetScanline( i );

                for ( uint32_t j = 0; j < pSrcBitmap->GetWidth(); ++j )
                {
                    for ( uint32_t k = 0; k < channelCount; ++k )
                    {
                        pSrcScanline[j * channelCount + k] = ChannelType( std::clamp( float(pSrcScanline[j]) * (maxValues[k] + diffLevels) / ( pDivisorScanline[j * channelCount + k] + diffLevels ), 0.0f, float( PixelFormatTraits<pixelFormat>::channelMax ) ) );
                    }
                }
            }
        } );
        this->_pDstBitmap = this->_pSrcBitmap;
    }
};

BitmapDivisor::BitmapDivisor( IBitmapPtr pSrcBitmap, IBitmapPtr pDivisor )
    : BaseTransform( pSrcBitmap )
    , _pDivisor( pDivisor )
{
}

std::shared_ptr<BitmapDivisor> BitmapDivisor::Create( IBitmapPtr pSrcBitmap, IBitmapPtr pDivisor )
{
    if ( !pSrcBitmap )
        throw std::invalid_argument( "pSrcBitmap is null" );

    if ( !pDivisor )
        throw std::invalid_argument( "pDivisor is null" );

    if ( pSrcBitmap->GetPixelFormat() != pDivisor->GetPixelFormat() )
        throw std::invalid_argument( "bitmaps should have the same pixel format" );

    if ( pSrcBitmap->GetWidth() != pDivisor->GetWidth() || pSrcBitmap->GetHeight() != pDivisor->GetHeight() )
        throw std::invalid_argument( "bitmaps should have the same size" );

    switch ( pSrcBitmap->GetPixelFormat() )
    {
        case PixelFormat::Gray8:
            return std::make_shared<BitmapDivisor_<PixelFormat::Gray8>>( pSrcBitmap, pDivisor );
        case PixelFormat::Gray16:
            return std::make_shared<BitmapDivisor_<PixelFormat::Gray16>>( pSrcBitmap, pDivisor );
        case PixelFormat::RGB24:
            return std::make_shared<BitmapDivisor_<PixelFormat::RGB24>>( pSrcBitmap, pDivisor );
        case PixelFormat::RGB48:
            return std::make_shared<BitmapDivisor_<PixelFormat::RGB48>>( pSrcBitmap, pDivisor );
        default:
            throw std::runtime_error( "pixel format must be known" );
    }
}

std::shared_ptr<BitmapDivisor> BitmapDivisor::Create( PixelFormat srcPixelFormat, IBitmapPtr pDivisor )
{
    if ( !pDivisor )
        throw std::invalid_argument( "pDivisor is null" );

    switch ( srcPixelFormat )
    {
        case PixelFormat::Gray8:
            return std::make_shared<BitmapDivisor_<PixelFormat::Gray8>>( nullptr, pDivisor );
        case PixelFormat::Gray16:
            return std::make_shared<BitmapDivisor_<PixelFormat::Gray16>>( nullptr, pDivisor );
        case PixelFormat::RGB24:
            return std::make_shared<BitmapDivisor_<PixelFormat::RGB24>>( nullptr, pDivisor );
        case PixelFormat::RGB48:
            return std::make_shared<BitmapDivisor_<PixelFormat::RGB48>>( nullptr, pDivisor );
        default:
            throw std::runtime_error( "pixel format must be known" );
    }
}

IBitmapPtr BitmapDivisor::Divide( IBitmapPtr pSrcBitmap, IBitmapPtr pDivisor )
{
    auto pSubtractor = Create( pSrcBitmap, pDivisor );
    return pSubtractor->RunAndGetBitmap();
}

ACMB_NAMESPACE_END