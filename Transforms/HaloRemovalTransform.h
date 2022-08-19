#pragma once

#include "basetransform.h"
#include "./../Core/IParallel.h"
#include "./../Tools/mathtools.h"

class BaseHaloRemovalTransform : public BaseTransform
{
protected:
    float _intensity;
    float _peakHue;
    float _sigma;
    float _bgL;

    BaseHaloRemovalTransform( IBitmapPtr pSrcBitmap, float intensity, float bgL, float peakHue, float sigma );

public:    
    static std::shared_ptr<BaseHaloRemovalTransform> Create( IBitmapPtr pSrcBitmap, float intensity, float bgL = 0.3f, float peakHue = 285.0f, float sigma = 40.0f );
    static IBitmapPtr RemoveHalo( IBitmapPtr pSrcBitmap, float intensity, float bgL = 0.3f, float peakHue = 285.0f, float sigma = 40.f );
    static IBitmapPtr AutoRemove( IBitmapPtr pSrcBitmap, float intensity );
};

template <PixelFormat pixelFormat>
class HaloRemovalTransform final: public BaseHaloRemovalTransform, public IParallel
{
    using ChannelType = typename PixelFormatTraits<pixelFormat>::ChannelType;   

public:
    
    HaloRemovalTransform( std::shared_ptr <Bitmap<pixelFormat>> pSrcBitmap, float intensity, float bgL, float peakHue, float sigma )
    : BaseHaloRemovalTransform( pSrcBitmap, intensity, bgL, peakHue, sigma )
    , IParallel( pSrcBitmap->GetHeight() )
    {
    }

    void Job( uint32_t i ) override
    {
        auto pSrcBitmap = std::static_pointer_cast< Bitmap<pixelFormat> >( _pSrcBitmap );

        const float a = ( 4 * _intensity ) / ( 4 * _bgL - ( 1 + _bgL ) * ( 1 + _bgL ) );
        const float b = -a * (1 + _bgL );
        const float c = a * _bgL;

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
               
                //hsl[2] *= ( 1 - _intensity * normalDist( hue, _peakHue, 1, _sigma ) );
                auto coef = a * hsl[2] * hsl[2] + b * hsl[2] + c;
                if ( hsl[2] > _bgL )
                {
                    //auto coef = ( hsl[2] - bgL ) / ( 1 - bgL ) * 0.5f;
                   
                    
                    //auto coef = normalDist( hsl[2], ( 1 + bgL ) / 2, _intensity, ( 1 - bgL ) / 6 );
                    hsl[2] = std::max(_bgL, hsl[2] * (1 - _intensity * hsl[1] *coef * normalDist(hue, _peakHue, 1, _sigma)));
                }
                /*else
                {
                    hsl[2] = std::min( _bgL, hsl[2] * ( 1 + _intensity * hsl[1] * coef * normalDist( hue, _peakHue, 1, _sigma ) ) );
                }*/

                hsl[1] *= ( 1 - _intensity * normalDist( hue, _peakHue, 1, _sigma ) );
                HslToRgb( hsl, rgb );
            }
        }
    }

    void Run() override
    {
        DoParallelJobs();
        _pDstBitmap = _pSrcBitmap;
    }
};
