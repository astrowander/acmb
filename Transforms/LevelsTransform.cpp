#include "LevelsTransform.h"
#include "ChannelEqualizer.h"

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

        _pCommonEqualizer = ChannelEqualizer::Create( pSrcBitmap,
                                                      {
                                                          [&] ( float val ) { return computePixelValue( val, levels.levels[0] ); },
                                                          [&] ( float val ) { return computePixelValue( val, levels.levels[0] ); },
                                                          [&] ( float val ) { return computePixelValue( val, levels.levels[0] ); }
                                                      } );

        if ( levels.adjustChannels )
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
        if ( levels.levels[i].min > levels.levels[i].max )
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

ACMB_NAMESPACE_END