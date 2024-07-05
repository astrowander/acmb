#include "MedianBlurTransform.h"

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

ACMB_NAMESPACE_BEGIN

template<PixelFormat pixelFormat>
class MedianBlurTransform_ : public MedianBlurTransform
{
    using ChannelType = typename PixelFormatTraits<pixelFormat>::ChannelType;
    static constexpr uint32_t channelCount = PixelFormatTraits<pixelFormat>::channelCount;
    static constexpr ChannelType channelMax = PixelFormatTraits<pixelFormat>::channelMax;

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
                auto pDstScanline = pDstBitmap->GetScanline( i );

                const int top = std::max( 0, i - ( _kernelSize - 1 ) / 2 );
                const int bottom = std::min<int>( _pSrcBitmap->GetHeight() - 1, i + _kernelSize / 2 );

                for ( int j = 0; j < int( _pSrcBitmap->GetWidth() ); ++j )
                {
                    const int left = std::max( 0, j - ( _kernelSize - 1 ) / 2 );
                    const int right = std::min<int>( _pSrcBitmap->GetWidth() - 1, j + _kernelSize / 2 );

                    for ( int ch = 0; ch < channelCount; ++ch )
                    {
                        std::vector<ChannelType> values;
                        values.reserve( (right - left + 1) * (bottom - top + 1) );

                        for ( int y = top; y <= bottom; ++y )
                        {
                            auto pPixel = pSrcBitmap->GetScanline( y ) + left * channelCount + ch;
                            for ( int x = left; x <= right; ++x )
                            {
                                values.push_back( *pPixel );
                                pPixel += channelCount;
                            }
                        }

                        auto medianIt = values.begin() + (values.size() - 1) / 2;
                        std::nth_element( values.begin(), medianIt, values.end() );
                        *pDstScanline++ = *medianIt;
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
