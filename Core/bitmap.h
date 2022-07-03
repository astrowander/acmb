#ifndef BITMAP_H
#define BITMAP_H
#include "enums.h"
#include "imageparams.h"
#include "../Tools/mathtools.h"

#include <vector>
#include <memory>
#include <stdexcept>
#include <limits>

#undef max

class IBitmap : public ImageParams
{
public:

    virtual ~IBitmap() = default;

    virtual char* GetPlanarScanline(uint32_t i) = 0;
    virtual uint32_t GetChannel(uint32_t i, uint32_t j, uint32_t k) const = 0;
    virtual void SetChannel(uint32_t i, uint32_t j, uint32_t k, uint32_t value) = 0;
    virtual uint32_t GetByteSize() = 0;

    virtual float GetInterpolatedChannel(float x, float y, uint32_t ch) = 0;

    static std::shared_ptr<IBitmap> Create(const std::string& fileName);
    static std::shared_ptr<IBitmap> Create(uint32_t width, uint32_t height, PixelFormat pixelFormat);
    static void Save(std::shared_ptr<IBitmap> pBitmap, const std::string& fileName);
};

using IBitmapPtr = std::shared_ptr<IBitmap>;

template<PixelFormat pixelFormat>
class Bitmap : public IBitmap
{
    using ChannelType = typename PixelFormatTraits<pixelFormat>::ChannelType;
    using ColorType = typename PixelFormatTraits<pixelFormat>::ColorType;
    using EnumColorType = typename PixelFormatTraits<pixelFormat>::EnumColorType;

    static constexpr auto channelCount = PixelFormatTraits<pixelFormat>::channelCount;

    std::vector<ChannelType> _data;

    void Fill(ColorType fillColor)
    {
        ChannelType channels[channelCount];
        for (uint32_t i = 0; i < channelCount; ++i)
        {
            auto channel = static_cast<ChannelType>(fillColor >> (BitsPerChannel(pixelFormat) * i));
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

    Bitmap(uint32_t width, uint32_t height, EnumColorType fillColor)
    : Bitmap(width, height, static_cast<ColorType>(fillColor))
    {
    }

    ChannelType* GetScanline(uint32_t i)
    {
        return &_data[_width * i * channelCount];
    }

    char* GetPlanarScanline(uint32_t i) override
    {
        return reinterpret_cast<char*>(&_data[_width * i * channelCount]);
    }

    uint32_t GetChannel(uint32_t i, uint32_t j, uint32_t k) const override
    {
        return _data[(_width * i + j) * channelCount + k];
    }

    void SetChannel(uint32_t i, uint32_t j, uint32_t k, uint32_t value) override
    {
        _data[(_width * i + j) * channelCount + k] = value;
    }

    uint32_t GetByteSize() override
    {
        return _width * _height * BytesPerPixel(pixelFormat);
    }

    const auto& GetData()
    {
        return _data;
    }

    void SetData(const std::vector<ChannelType>& data)
    {
        _data = data;
    }

    float GetInterpolatedChannel(float x, float y, uint32_t ch) override
    {
        if (x < 0 || x > _width - 1)
            throw std::invalid_argument("x");

        if (y < 0 || y > _height - 1)
            throw std::invalid_argument("y");

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

        const ChannelType maxChannel = std::numeric_limits<ChannelType>::max();
        return FitToBounds<float>(QuadraticInterpolation(y - y0, yIn[0], yIn[1], yIn[2]), 0, maxChannel);
    }
};



#endif // BITMAP_H
