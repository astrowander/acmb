#include "DeflickerTransform.h"
#include "converter.h"
#include "HistogramBuilder.h"
#include "LevelsTransform.h"
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include <algorithm>

using namespace oneapi::tbb;

ACMB_NAMESPACE_BEGIN

template<PixelFormat pixelFormat>
class DeflickerTransformImpl : public DeflickerTransform
{
    using ChannelType = typename PixelFormatTraits<pixelFormat>::ChannelType;
    static constexpr auto channelCount = PixelFormatTraits<pixelFormat>::channelCount;
    static constexpr auto channelMax = PixelFormatTraits<pixelFormat>::channelMax;

    static constexpr PixelFormat cGrayFormat = ConstructPixelFormat( BitsPerChannel( pixelFormat ), 1 );

public:
    DeflickerTransformImpl( const Settings& settings )
    : DeflickerTransform( settings )
    {
    }

    virtual void Run() override
    {
        for ( int i = 0; i < _settings.iterations; ++i )
        {
            RunOneIteration();
        }
    }

    void RunOneIteration()
    {
        std::vector<float> medians;
        medians.reserve( _settings.bitmaps.size() );

        for ( auto pBitmap : _settings.bitmaps )
        {
            auto pSrcBitmap = std::static_pointer_cast< Bitmap<pixelFormat> >( pBitmap );
            auto pGrayBitmap = Converter::Convert( pSrcBitmap, cGrayFormat );
            auto pHistBuilder = HistogramBuilder::Create( pGrayBitmap );
            pHistBuilder->BuildHistogram();
            medians.push_back( log( pHistBuilder->GetChannelStatistics( 0 ).median ) );
        }

        auto mediansCopy = medians;
        auto medianIt = mediansCopy.begin() + mediansCopy.size() / 2;
        std::nth_element( mediansCopy.begin(), medianIt, mediansCopy.end() );

        const float targetMedian = float( *medianIt );
        
        for ( size_t i = 0; i < _settings.bitmaps.size(); ++i )
        {
            std::cout << "Start " << i << "-th frame: " << std::endl;
            auto pBitmap = _settings.bitmaps[i];
            auto pSrcBitmap = std::static_pointer_cast< Bitmap<pixelFormat> >(pBitmap);
            const float dI = targetMedian - medians[i];
            if ( fabs( dI ) < std::numeric_limits<float>::epsilon() )
                continue;

            const float mult = exp( dI );

            parallel_for( blocked_range<uint32_t>( 0, pSrcBitmap->GetHeight() ), [&] ( const blocked_range<uint32_t>& range )
            {
                for ( uint32_t y = range.begin(); y != range.end(); ++y )
                {
                    auto pScanline = pSrcBitmap->GetScanline( y );
                    for ( uint32_t x = 0; x < pSrcBitmap->GetWidth(); ++x )
                    {
                        for ( uint32_t ch = 0; ch < channelCount; ++ch )
                        {
                            pScanline[x * channelCount + ch] = ChannelType( std::clamp<float>( pScanline[x * channelCount + ch] * mult, 0, channelMax ) + 0.5f );
                        }
                    }
                }
            } );
        }
    }
};

DeflickerTransform::DeflickerTransform( const Settings& settings )
: _settings( settings )
{
}

std::shared_ptr<DeflickerTransform> DeflickerTransform::Create( const Settings& settings )
{
    switch ( settings.bitmaps[0]->GetPixelFormat() )
    {
    case PixelFormat::RGB24:
        return std::make_shared<DeflickerTransformImpl<PixelFormat::RGB24>>( settings );
    case PixelFormat::Gray8:
        return std::make_shared<DeflickerTransformImpl<PixelFormat::Gray8>>( settings );
    case PixelFormat::RGB48:
        return std::make_shared<DeflickerTransformImpl<PixelFormat::RGB48>>( settings );
    case PixelFormat::Gray16:
        return std::make_shared<DeflickerTransformImpl<PixelFormat::Gray16>>( settings );
    default:
        throw std::invalid_argument( "DeflickerTransform: unsupported pixel format" );
    }
}

void DeflickerTransform::Deflicker( const Settings& settings )
{
    auto pDeflickerTransform = Create( settings );
    pDeflickerTransform->Run();
}

ACMB_NAMESPACE_END