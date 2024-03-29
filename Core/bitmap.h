#pragma once
#include "enums.h"
#include "IPipelineElement.h"
#include "../Tools/mathtools.h"
#include "camerasettings.h"

#include <vector>
#include <memory>
#include <stdexcept>
#include <limits>
#include <algorithm>

//#undef max

ACMB_NAMESPACE_BEGIN

/// <summary>
/// Abstract factory class for bitmap of any pixel format
/// </summary>
class IBitmap : public IPipelineFirstElement
{
public:

    virtual ~IBitmap() = default;
    /// returns given scanline casted to char*
    virtual char* GetPlanarScanline(uint32_t i) = 0;
    /// returns k-th channel of j-th pixel on i-th scanline
    virtual uint32_t GetChannel(uint32_t i, uint32_t j, uint32_t k) const = 0;
    /// sets k-th channel of j-th pixel on i-th scanline
    virtual void SetChannel(uint32_t i, uint32_t j, uint32_t k, uint32_t value) = 0;
    /// returns count of allocated bytes
    virtual uint32_t GetByteSize() const = 0;
    /// returns deep copy of a bitmap
    virtual std::shared_ptr<IBitmap> Clone() const = 0;

    /// creates bitmap from a given file
    static std::shared_ptr<IBitmap> Create( const std::string& fileName, PixelFormat outputFormat = PixelFormat::Unspecified );
    /// creates bitmap from a given file
    static std::shared_ptr<IBitmap> Create( const std::shared_ptr<std::istream> pStream, PixelFormat outputFormat = PixelFormat::Unspecified );
    /// creates bitmap with given params
    static std::shared_ptr<IBitmap> Create(uint32_t width, uint32_t height, PixelFormat pixelFormat);
    /// saves given bitmap to a file
    static void Save(std::shared_ptr<IBitmap> pBitmap, const std::string& fileName);
};
/// alias for pointer to bitmap
using IBitmapPtr = std::shared_ptr<IBitmap>;

/// represents bitmap of a certain pixel format
template<PixelFormat pixelFormat>
class Bitmap : public IBitmap, public std::enable_shared_from_this<Bitmap<pixelFormat>>
{
    /// alias for channel type (uint8_t/uint16_t)
    using ChannelType = typename PixelFormatTraits<pixelFormat>::ChannelType;
    /// alias for color type (uint32_t/uint64_t)
    using ColorType = typename PixelFormatTraits<pixelFormat>::ColorType;
    /// alias for enum with predefined color
    using EnumColorType = typename PixelFormatTraits<pixelFormat>::EnumColorType;
    /// alias for channel count of pixel format
    static constexpr auto channelCount = PixelFormatTraits<pixelFormat>::channelCount;

    std::vector<ChannelType> _data;

    void Fill(ColorType fillColor)
    {
        ChannelType channels[channelCount];
        for (uint32_t i = 0; i < channelCount; ++i)
        {
            auto channel = static_cast<ChannelType>(fillColor >> (BitsPerChannel(pixelFormat) * i) & PixelFormatTraits<pixelFormat>::channelMax);
            channels[channelCount - i - 1] = channel;
        }

        for (uint32_t i = 0; i < _height; ++i)
        {
            for (uint32_t j = 0; j < _width; ++j)
            {
                for (uint32_t k = 0; k < channelCount; ++k)
                {
                    _data[(i * _width + j) * channelCount + k] = channels[k];
                }
            }
        }
    }

public:
    /// creates bitmap with given size and fills it with given color (black by default)
    Bitmap(uint32_t width, uint32_t height, ColorType fillColor = 0)
    {
        if (width == 0 || height == 0)
            throw std::invalid_argument("size should not be zero");

        if (width > 0xFFFF || height > 0xFFFF)
            throw std::invalid_argument("size  is too large");

        _width = width;
        _height = height;
        _pixelFormat = pixelFormat;
        _data.resize(_width * _height * channelCount);

        if (fillColor != 0)
        {
            Fill(fillColor);
        }
    }
    /// creates bitmap with given size and fills it with given color
    Bitmap(uint32_t width, uint32_t height, EnumColorType fillColor)
    : Bitmap(width, height, static_cast<ColorType>(fillColor))
    {
    }
    /// returns given scanline
    ChannelType* GetScanline(uint32_t i)
    {
        return &_data[_width * i * channelCount];
    }
    /// returns given scanline casted to char*
    char* GetPlanarScanline(uint32_t i) override
    {
        return reinterpret_cast<char*>(&_data[_width * i * channelCount]);
    }
    /// returns k-th channel of j-th pixel on i-th scanline
    uint32_t GetChannel(uint32_t i, uint32_t j, uint32_t k) const override
    {
        return _data[(_width * i + j) * channelCount + k];
    }
    /// sets k-th channel of j-th pixel on i-th scanline
    void SetChannel(uint32_t i, uint32_t j, uint32_t k, uint32_t value) override
    {
        _data[(_width * i + j) * channelCount + k] = value;
    }
    /// returns count of allocated bytes
    uint32_t GetByteSize() const override
    {
        return _width * _height * BytesPerPixel(pixelFormat);
    }
    
    /// returns data vector    
    const auto& GetData() const
    {
        return _data;
    }

    /// returns data vector  UNSAFE 
    auto& GetData()
    {
        return _data;
    }
    /// sets data vector
    void SetData(const std::vector<ChannelType>& data)
    {
        if ( data.size() != _data.size() )
            throw std::invalid_argument( "data has different size" );

        _data = data;
    }
    /// receives arbitrary coords, returns interpolated channel value
    float GetInterpolatedChannel(float x, float y, uint32_t ch) const
    {
        if (x < 0 || x > _width - 1)
            return 0.0f;

        if (y < 0 || y > _height - 1)
            return 0.0f;

        if (ch >= channelCount)
            throw std::invalid_argument("ch");

        uint32_t x0 = FastRound<uint32_t>(x);
        if (x0 == _width - 1)
            x0 -= 2;
        else if (x0 >= 1)
            x0 -= 1;

        uint32_t y0 = FastRound<uint32_t>(y);
        if (y0 == _height - 1)
            y0 -= 2;
        else if (y0 >= 1)
            y0 -= 1;

        float yIn[3] =
        {
            QuadraticInterpolation(x - x0, _data[(y0 * _width + x0) * channelCount + ch], _data[(y0 * _width + x0 + 1) * channelCount + ch], _data[(y0 * _width + x0 + 2) * channelCount + ch]),
            QuadraticInterpolation(x - x0, _data[((y0 + 1) * _width + x0) * channelCount + ch], _data[((y0 + 1) * _width + x0 + 1) * channelCount + ch], _data[((y0 + 1) * _width + x0 + 2) * channelCount + ch]),
            QuadraticInterpolation(x - x0, _data[((y0 + 2) * _width + x0) * channelCount + ch], _data[((y0 + 2) * _width + x0 + 1) * channelCount + ch], _data[((y0 + 2) * _width + x0 + 2) * channelCount + ch])
        };

        return std::clamp<float>(QuadraticInterpolation(y - y0, yIn[0], yIn[1], yIn[2]), 0, std::numeric_limits<ChannelType>::max() );
    }

    virtual IBitmapPtr ProcessBitmap( std::shared_ptr<IBitmap> ) override
    {
        return this->shared_from_this();
    }

    virtual std::shared_ptr<IBitmap> Clone() const override
    {
        auto pRes = std::make_shared<Bitmap<pixelFormat>>( _width, _height );
        pRes->SetData( _data );
        return pRes;
    }
};

ACMB_NAMESPACE_END
