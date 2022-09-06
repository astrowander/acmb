#pragma once

#include "../imagedecoder.h"
#include "../../Core/bitmap.h"
#include "tinytiffreader.hxx"
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

struct TinyTIFFReaderFile;
class TiffDecoder : public ImageDecoder
{
    TinyTIFFReaderFile* _pReader;

    template <PixelFormat pixelFormat>
    void JoinChannels( std::shared_ptr<Bitmap<pixelFormat>> pBitmap, const std::vector<uint8_t>& data )
    {
        constexpr auto channelCount = PixelFormatTraits<pixelFormat>::channelCount;
        auto pData = pBitmap->GetScanline( 0 );

        oneapi::tbb::parallel_for( oneapi::tbb::blocked_range<int>( 0, _height ), [&] ( const oneapi::tbb::blocked_range<int>& range )
        {
            for ( int i = range.begin(); i < range.end(); ++i )
            {
                for ( uint32_t j = 0; j < _width; ++j )
                {
                    for ( uint16_t ch = 0; ch < channelCount; ++ch )
                    {
                        pData[( i * _width + j ) * channelCount + ch] = data[ch * _width * _height + i * _width + j];
                    }
                }
            }
        } );
    }

public:
    TiffDecoder() = default;

    void Attach( const std::string& fileName ) override;
    void Attach( std::shared_ptr<std::istream> pStream ) override;
    void Detach() override;

    std::shared_ptr<IBitmap> ReadBitmap() override;
    std::shared_ptr<IBitmap> ReadStripe( uint32_t stripeHeight = 0 ) override;

    uint32_t GetCurrentScanline() const override;

    static std::unordered_set <std::string> GetExtensions()
    {
        return { ".tiff", ".tif" };
    }

    ADD_EXTENSIONS
};