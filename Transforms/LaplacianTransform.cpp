#include "LaplacianTransform.h"

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

ACMB_NAMESPACE_BEGIN

template<PixelFormat pixelFormat>
class ConvolutionTransform_ : public ConvolutionTransform
{
    using ChannelType = typename PixelFormatTraits<pixelFormat>::ChannelType;
    static constexpr uint32_t cChannelCount = PixelFormatTraits<pixelFormat>::channelCount;
    static constexpr ChannelType cChannelMax = PixelFormatTraits<pixelFormat>::channelMax;

public:
    ConvolutionTransform_( IBitmapPtr pSrcBitmap, const Settings& settings )
    : ConvolutionTransform(pSrcBitmap, settings)
    {
    }

    virtual void Run() override
    {
        auto pSrcBitmap = std::static_pointer_cast< Bitmap<pixelFormat> >(_pSrcBitmap);
        auto pDstBitmap = std::make_shared<Bitmap<pixelFormat>>( _pSrcBitmap->GetWidth() - _settings.kernelSize.width + 1, 
                                                                 _pSrcBitmap->GetHeight() - _settings.kernelSize.height + 1 );

        const uint32_t srcWidth = pSrcBitmap->GetWidth();

        tbb::parallel_for( tbb::blocked_range<int>( 0, pDstBitmap->GetHeight() ), [&] ( const tbb::blocked_range<int>& range )
        {
            for ( int i = range.begin(); i < range.end(); ++i )
            {
                auto pSrcScanline = pSrcBitmap->GetScanline( i );
                auto pDstScanline = pDstBitmap->GetScanline( i );
                for ( uint32_t j = 0; j < pDstBitmap->GetWidth(); ++j )
                {
                    float sum = 0;
                    for ( int y = 0; y < _settings.kernelSize.height; ++y )
                    {
                        for ( int x = 0; x < _settings.kernelSize.width; ++x )
                        {
                            sum += pSrcScanline[ y * srcWidth + j + x] * _settings.kernel[y * _settings.kernelSize.width + x];
                        }
                    }

                    pDstScanline[j] = static_cast<ChannelType>(std::clamp<float>(abs( sum ), 0, cChannelMax));
                }
            }
        } );

        _pDstBitmap = pDstBitmap;
    }

    virtual void ValidateSettings() override
    {
        if ( _settings.kernelSize.width < 1 || _settings.kernelSize.height < 1 )
            throw std::invalid_argument( "kernel size must be >= 1" );

        if ( _settings.kernel.size() != _settings.kernelSize.width * _settings.kernelSize.height )
            throw std::invalid_argument( "kernel size must be equal to kernel width * kernel height" );
    }
};

ConvolutionTransform::ConvolutionTransform( IBitmapPtr pSrcBitmap, const Settings& settings )
: BaseTransform( pSrcBitmap )
, _settings( settings )
{
}

std::shared_ptr<ConvolutionTransform> ConvolutionTransform::Create( IBitmapPtr pSrcBitmap, const Settings& settings )
{
    if ( !pSrcBitmap )
        throw std::invalid_argument( "pSrcBitmap is null" );
    if ( settings.kernelSize.width < 1 || settings.kernelSize.height < 1 )
        throw std::invalid_argument( "kernel size must be >= 1" );
    if ( settings.kernel.size() != settings.kernelSize.width * settings.kernelSize.height )
        throw std::invalid_argument( "kernel size must be equal to kernel width * kernel height" );

    switch ( pSrcBitmap->GetPixelFormat() )
    {
        case PixelFormat::Gray8:
            return std::make_shared<ConvolutionTransform_<PixelFormat::Gray8>>( pSrcBitmap, settings );
        case PixelFormat::Gray16:
            return std::make_shared<ConvolutionTransform_<PixelFormat::Gray16>>( pSrcBitmap, settings );
        default:
            throw std::runtime_error( "ConvolutionTransform supports only grayscale images" );
    }
}

std::shared_ptr<ConvolutionTransform> ConvolutionTransform::Create( PixelFormat pixelFormat, const Settings& settings )
{
    switch ( pixelFormat )
    {
        case PixelFormat::Gray8:
            return std::make_shared<ConvolutionTransform_<PixelFormat::Gray8>>( nullptr, settings );
        case PixelFormat::Gray16:
            return std::make_shared<ConvolutionTransform_<PixelFormat::Gray16>>( nullptr, settings );
        default:
            throw std::runtime_error( "ConvolutionTransform supports only grayscale images" );
    }
}

IBitmapPtr ConvolutionTransform::ApplyConvolution( IBitmapPtr pSrcBitmap, const Settings& settings )
{
    auto pTransform = Create( pSrcBitmap, settings );
    return pTransform->RunAndGetBitmap();
}

IBitmapPtr ConvolutionTransform::ApplyLaplacian( IBitmapPtr pSrcBitmap )
{
    Settings settings{.kernelSize = {5, 5},
                       .kernel = { 0,  0, -1,  0, 0,
                                   0, -1, -2, -1, 0,
                                  -1, -2, 16, -2, -1,
                                   0, -1, -2, -1,  0,
                                   0,  0, -1,  0,  0 }
    };

    auto pTransform = Create( pSrcBitmap, settings );
    return pTransform->RunAndGetBitmap();
}

void ConvolutionTransform::CalcParams( std::shared_ptr<ImageParams> pParams )
{
    _width = pParams->GetWidth() - _settings.kernelSize.width + 1;
    _height = pParams->GetHeight() - _settings.kernelSize.height + 1;
    _pixelFormat = pParams->GetPixelFormat();
}

ACMB_NAMESPACE_END