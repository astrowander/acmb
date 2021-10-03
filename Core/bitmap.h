#ifndef BITMAP_H
#define BITMAP_H
#include "enums.h"
#include "imageparams.h"

#include <type_traits>
#include <vector>

#include <memory>

class IBitmap : public ImageParams
{
public:

    virtual ~IBitmap() = default;

    virtual char* GetPlanarScanline(uint32_t i) = 0;
    virtual uint32_t GetChannel(uint32_t i, uint32_t j, uint32_t k) const = 0;
    virtual void SetChannel(uint32_t i, uint32_t j, uint32_t k, uint32_t value) = 0;
    virtual uint32_t GetByteSize() = 0;

    static std::shared_ptr<IBitmap> Create(const std::string& fileName);
    static void Save(std::shared_ptr<IBitmap> pBitmap, const std::string& fileName);
};



template<PixelFormat pixelFormat>
class Bitmap : public IBitmap
{
    using ChannelType = typename std::conditional_t<((uint32_t)pixelFormat >> 16) == 1, uint8_t, uint16_t>;
    using ColorType = typename std::conditional_t<((uint32_t)pixelFormat >> 16) == 1, uint32_t, uint64_t>;
    using EnumColorType = typename std::conditional_t<((uint32_t)pixelFormat >> 16) == 1, ARGB32Color, ARGB64Color>;
    static constexpr auto channelCount = ChannelCount(pixelFormat);

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
};



#endif // BITMAP_H
