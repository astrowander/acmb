#include "LevelsTransform.h"
#include "ChannelEqualizer.h"
#include "HistogramBuilder.h"

ACMB_NAMESPACE_BEGIN

template<PixelFormat pixelFormat>
class LevelsTransformImpl : public LevelsTransform
{
    using ChannelType = typename PixelFormatTraits<pixelFormat>::ChannelType;
    static constexpr uint32_t channelCount = PixelFormatTraits<pixelFormat>::channelCount;
    static constexpr ChannelType channelMax = PixelFormatTraits<pixelFormat>::channelMax;

public:

    LevelsTransformImpl( IBitmapPtr pSrcBitmap, const Settings& levels )
    : LevelsTransform( pSrcBitmap )
    {
        const auto computePixelValue = [] ( float srcVal, const ChannelLevels& channelLevels )
        {
            if ( srcVal > 0.99f )
                srcVal = srcVal;

            if ( srcVal <= channelLevels.min )
                return 0.0f;
            if ( srcVal >= channelLevels.max )
                return 1.0f;

            const float res = (srcVal - channelLevels.min) / (channelLevels.max - channelLevels.min);
            return std::clamp<float>( std::pow( res, 1.0f / channelLevels.gamma ), 0.0f, 1.0f );
        };

        if constexpr ( channelCount == 3)
        { 
        _pCommonEqualizer = ChannelEqualizer::Create( pSrcBitmap,
                                                      {
                                                          [&] ( float val ) { return computePixelValue( val, levels.levels[0] ); },
                                                          [&] ( float val ) { return computePixelValue( val, levels.levels[0] ); },
                                                          [&] ( float val ) { return computePixelValue( val, levels.levels[0] ); }
                                                      } );
        }
        else if constexpr ( channelCount == 1 )
        {
            _pCommonEqualizer = ChannelEqualizer::Create( pSrcBitmap,
                                                          {
                                                              [&] ( float val ) { return computePixelValue( val, levels.levels[0] ); }
                                                          } );
        }
        

        if ( levels.adjustChannels && channelCount == 3 )
        {
            _pPerChannelEqualizer = ChannelEqualizer::Create( pSrcBitmap,
                                                             {
                                                                 [&] ( float val ) { return computePixelValue( val, levels.levels[1] ); },
                                                                 [&] ( float val ) { return computePixelValue( val, levels.levels[2] ); },
                                                                 [&] ( float val ) { return computePixelValue( val, levels.levels[3] ); }
                                                             } );
        }
    }

    virtual void Run() override
    {
        _pDstBitmap = _pCommonEqualizer->RunAndGetBitmap();
        if ( _pPerChannelEqualizer )
        {
            _pPerChannelEqualizer->SetSrcBitmap( _pDstBitmap );
            _pDstBitmap = _pPerChannelEqualizer->RunAndGetBitmap();
        }
    }

    virtual void ValidateSettings() override
    {     
    }

};

LevelsTransform::LevelsTransform( IBitmapPtr pSrcBitmap )
: BaseTransform( pSrcBitmap )
{
}

std::shared_ptr<LevelsTransform> LevelsTransform::Create( IBitmapPtr pSrcBitmap, const Settings& levels )
{
    if ( !pSrcBitmap )
        throw std::invalid_argument( "pSrcBitmap is null" );

    for ( uint32_t i = 0; i < levels.levels.size(); ++i )
    {
        if ( levels.levels[i].min >= levels.levels[i].max )
            throw std::invalid_argument( "min > max" );

        if ( levels.levels[i].min < 0.0f || levels.levels[i].max > 1.0f )
            throw std::invalid_argument( "min < 0.0f || max > 1.0f" );

        if ( levels.levels[i].gamma <= 0.0f )
            throw std::invalid_argument( "gamma < 0.0f" );
    }

    switch ( pSrcBitmap->GetPixelFormat() )
    {
        case PixelFormat::Gray8:
            return std::make_shared<LevelsTransformImpl<PixelFormat::Gray8>>( pSrcBitmap, levels );
        case PixelFormat::Gray16:
            return std::make_shared<LevelsTransformImpl<PixelFormat::Gray16>>( pSrcBitmap, levels );
        case PixelFormat::RGB24:
            return std::make_shared<LevelsTransformImpl<PixelFormat::RGB24>>( pSrcBitmap, levels );
        case PixelFormat::RGB48:
            return std::make_shared<LevelsTransformImpl<PixelFormat::RGB48>>( pSrcBitmap, levels );
        default:
            throw std::invalid_argument( "LevelsTransform: Unsupported pixel format" );
    }
}

std::shared_ptr<LevelsTransform> LevelsTransform::Create( PixelFormat pixelFormat, const Settings& levels )
{
    return Create( IBitmap::Create( 1, 1, pixelFormat), levels);
}

IBitmapPtr LevelsTransform::ApplyLevels( IBitmapPtr pSrcBitmap, const Settings& levels )
{
    return LevelsTransform::Create( pSrcBitmap, levels )->RunAndGetBitmap();
}

LevelsTransform::Settings LevelsTransform::GetAutoSettings( IBitmapPtr pSrcBitmap, bool adjustChannels )
{
    auto pHistogramBuilder = HistogramBuilder::Create( pSrcBitmap );
    pHistogramBuilder->BuildHistogram();

    const auto colorSpace = GetColorSpace( pSrcBitmap->GetPixelFormat() );
    const auto pixelFormat = pSrcBitmap->GetPixelFormat();
    const auto bytesPerChannel = BytesPerChannel( pixelFormat );
    const float absoluteMax = (bytesPerChannel == 1) ? 255.0f : 65535.0f;

    constexpr float logTargetMedian = -2.14f;

    Settings result { .adjustChannels = adjustChannels };

    switch ( colorSpace )
    {
        case ColorSpace::Gray:
        {
            result.levels[0].min = pHistogramBuilder->GetChannelStatistics( 0 ).min / absoluteMax;
            result.levels[0].max = pHistogramBuilder->GetChannelStatistics( 0 ).max / absoluteMax;
            const float denom = log( (pHistogramBuilder->GetChannelStatistics( 0 ).median / absoluteMax - result.levels[0].min) / (result.levels[0].max - result.levels[0].min) );
            result.levels[0].gamma = denom / logTargetMedian;
            break;
        }
        case ColorSpace::RGB:
        {
            std::array<float, 3> channelMins = { pHistogramBuilder->GetChannelStatistics( 0 ).min / absoluteMax, pHistogramBuilder->GetChannelStatistics( 1 ).min / absoluteMax, pHistogramBuilder->GetChannelStatistics( 2 ).min / absoluteMax };
            std::array<float, 3> channelMaxs = { pHistogramBuilder->GetChannelStatistics( 0 ).max / absoluteMax, pHistogramBuilder->GetChannelStatistics( 1 ).max / absoluteMax, pHistogramBuilder->GetChannelStatistics( 2 ).max / absoluteMax };
            std::array<float, 3> channelMedians = { pHistogramBuilder->GetChannelStatistics( 0 ).median / absoluteMax, pHistogramBuilder->GetChannelStatistics( 1 ).median / absoluteMax, pHistogramBuilder->GetChannelStatistics( 2 ).median / absoluteMax };
            result.levels[0].min = std::min( { channelMins[0], channelMins[1], channelMins[2] } );
            result.levels[0].max = std::max( { channelMaxs[0], channelMaxs[1], channelMaxs[2] } );
            const float denom = log( (channelMedians[1] - channelMins[1]) / (channelMaxs[1] - channelMins[1]) );
            result.levels[0].gamma = denom / logTargetMedian;

            if ( adjustChannels )
            {
                const float range = result.levels[0].max - result.levels[0].min;
                result.levels[1].min = (channelMins[0] - result.levels[0].min) / range;
                result.levels[1].max = (channelMaxs[0] - result.levels[0].min) / range;
                result.levels[2].min = (channelMins[1] - result.levels[0].min) / range;
                result.levels[2].max = (channelMaxs[1] - result.levels[0].min) / range;
                result.levels[3].min = (channelMins[2] - result.levels[0].min) / range;
                result.levels[3].max = (channelMaxs[2] - result.levels[0].min) / range;

                result.levels[1].gamma = log( (channelMedians[0] - channelMins[0]) / (channelMaxs[0] - channelMins[0]) ) / denom;
                result.levels[3].gamma = log( (channelMedians[2] - channelMins[2]) / (channelMaxs[2] - channelMins[2]) ) / denom;
            }
            break;
        }
        default:
            throw std::invalid_argument( "LevelsTransform: unsupported color space" );
    }

    return result;
}

ACMB_NAMESPACE_END