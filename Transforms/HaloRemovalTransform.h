#pragma once

#include "basetransform.h"
#include "./../Core/IParallel.h"
#include "./../Tools/mathtools.h"

class BaseHaloRemovalTransform : public BaseTransform
{
protected:
    float _intensity;
    float _peakHue = 270;
    float _sigma = 15;

public:
    BaseHaloRemovalTransform( IBitmapPtr pSrcBitmap, float intensity );
    static std::shared_ptr<BaseHaloRemovalTransform> Create( IBitmapPtr pSrcBitmap, float intensity );
    static IBitmapPtr RemoveHalo( IBitmapPtr pSrcBitmap, float intensity );
};

template <PixelFormat pixelFormat>
class HaloRemovalTransform : public BaseHaloRemovalTransform, public IParallel
{
    using ChannelType = typename PixelFormatTraits<pixelFormat>::ChannelType;

public:
    HaloRemovalTransform( std::shared_ptr <Bitmap<pixelFormat>> pSrcBitmap, float intensity)
    : BaseHaloRemovalTransform(pSrcBitmap, intensity)
    , IParallel(pSrcBitmap->GetHeight())
    { }

    void Job( uint32_t i ) override
    {
        auto pSrcBitmap = std::static_pointer_cast< Bitmap<pixelFormat> >( _pSrcBitmap );
        for ( size_t j = 0; j < pSrcBitmap->GetWidth(); ++j )
        {
            auto rgb = std::span<ChannelType, 3>( pSrcBitmap->GetScanline( i ) + j * 3, 3 );
            auto hsl = RgbToHsl( rgb );
            
            const auto lbound = _peakHue - 3 * _sigma;
            const auto ubound = _peakHue + 3 * _sigma;

            float hue = hsl[0];
            
            if ( lbound < 0 && hsl[0] > lbound + 360 )
            {
                hue -= 360;

            }
            else if ( ubound > 360 && hsl[0] < ubound - 360 )
            {
                hue += 360;
            }

            if ( hue > lbound && hue < ubound  )
            {
                hsl[1] *= hsl[1] * normalDist(hue, _peakHue, 1 - _intensity, _sigma);
                hsl[2] *= std::min(1.0f, 1 + normalDist( hue, _peakHue, 0.1, 5 ));
                HslToRgb( hsl, rgb );
            }
        }
    }

    void Run()
    {
        DoParallelJobs();
        _pDstBitmap = _pSrcBitmap;
    }
};
