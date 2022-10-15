#include "BitmapDivisor.h"
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>
#include <algorithm>
#include <array>

ACMB_NAMESPACE_BEGIN

template <PixelFormat pixelFormat>
class BitmapDivisor_ final : public BitmapDivisor
{
    using ChannelType = typename PixelFormatTraits<pixelFormat>::ChannelType;

public:

    BitmapDivisor_( IBitmapPtr pSrcBitmap, IBitmapPtr pDivisor )
    : BitmapDivisor( pSrcBitmap, pDivisor )
    {
    }

    virtual void Run() override
    {
        auto pSrcBitmap = std::static_pointer_cast< Bitmap<pixelFormat> >( _pSrcBitmap );
        auto pDivisor = std::static_pointer_cast< Bitmap<pixelFormat> >( _pDivisor );
        constexpr uint32_t channelCount = PixelFormatTraits<pixelFormat>::channelCount;
        using ChannelType = typename PixelFormatTraits<pixelFormat>::ChannelType;

        const auto width = pSrcBitmap->GetWidth();
        const auto height = pSrcBitmap->GetHeight();
        const auto halfWidth = width / 2 + width % 2;
        const auto halfHeight = height / 2 + height % 2;

        std::array<std::vector<ChannelType>, channelCount * 4> subpixelVectors;
        for ( auto& subpixelVector : subpixelVectors )
            subpixelVector.resize( halfWidth * halfHeight );

        oneapi::tbb::parallel_for( oneapi::tbb::blocked_range<int>( 0, height ), [&] ( const oneapi::tbb::blocked_range<int>& range )
        {
            for ( int y = range.begin(); y < range.end(); ++y )
            {
                auto pDivisorScanline = pDivisor->GetScanline( y );

                for ( uint32_t x = 0; x < pDivisor->GetWidth(); ++x )
                {
                    const auto subpixelIndex = (x % 2) + (2 * (y % 2) );
                    for ( uint32_t ch = 0; ch < channelCount; ++ch )
                    {
                        subpixelVectors[4 * ch + subpixelIndex][halfWidth * ( y / 2 ) + ( x / 2 )] = pDivisorScanline[x * channelCount + ch];
                    }
                }
            }
        } );

#ifdef NDEBUG
        for ( auto& subpixelVector : subpixelVectors )
            oneapi::tbb::parallel_sort( subpixelVector.begin(), subpixelVector.end() );
#else
        for ( auto& subpixelVector : subpixelVectors )
            std::sort( subpixelVector.begin(), subpixelVector.end() );
#endif
        std::array <ChannelType, channelCount * 4> maxValues;
        for ( int i = 0; i < channelCount * 4; ++i )
        {
            maxValues[i] = subpixelVectors[i][ size_t( subpixelVectors[i].size() * 0.995f )];
        }

        const float srcBlackLevel = pSrcBitmap->GetCameraSettings() ? pSrcBitmap->GetCameraSettings()->blackLevel : 0;

        oneapi::tbb::parallel_for( oneapi::tbb::blocked_range<int>( 0, _pSrcBitmap->GetHeight() ), [&] ( const oneapi::tbb::blocked_range<int>& range )
        {
            for ( int y = range.begin(); y < range.end(); ++y )
            {
                auto pSrcScanline = pSrcBitmap->GetScanline( y );
                auto pDivisorScanline = pDivisor->GetScanline( y );

                for ( uint32_t x = 0; x < pSrcBitmap->GetWidth(); ++x )
                {
                    const auto subpixelIndex = ( x % 2 ) + ( 2 * ( y % 2 ) );
                    for ( uint32_t ch = 0; ch < channelCount; ++ch )
                    {
                        const size_t index = x * channelCount + ch;
                        const float srcValue = pSrcScanline[index];
                        const float divisorValue = pDivisorScanline[index];
                        float coeff = ( maxValues[4 * ch + subpixelIndex] - srcBlackLevel ) / ( divisorValue - srcBlackLevel );
                        coeff = 1.0f + ( coeff - 1.0f ) * 0.25f;
                        const float maxChannel = pSrcBitmap->GetCameraSettings() ? float (pSrcBitmap->GetCameraSettings()->maxChannel) : float(PixelFormatTraits<pixelFormat>::channelMax );
                        const float res = std::min( srcBlackLevel + std::max(0.0f,  srcValue - srcBlackLevel ) * coeff,  maxChannel );
                        pSrcScanline[index] = ChannelType( res );
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